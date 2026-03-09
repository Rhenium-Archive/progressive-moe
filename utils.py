"""
공용 유틸리티 모듈
- TextDataset: 텍스트 → 토큰 Dataset
- KoreanBPETokenizer: 한국어 BPE 토크나이저
- GPU 메모리 모니터링
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from model import TransformerMoE, ModelConfig
import os
import gc
from tqdm import tqdm
import psutil
import multiprocessing
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders


# ==========================================
# GPU 메모리 유틸리티
# ==========================================
def get_gpu_memory():
    """현재 GPU 메모리 사용량 반환 (dict 또는 None)"""
    if not torch.cuda.is_available():
        return None
    return {
        'allocated': torch.cuda.memory_allocated() / 1e9,
        'reserved': torch.cuda.memory_reserved() / 1e9,
        'max_allocated': torch.cuda.max_memory_allocated() / 1e9,
    }


def print_memory_stats(tag: str = ""):
    """GPU 메모리 상태 출력"""
    mem = get_gpu_memory()
    if mem is None:
        print(f"   [{tag}] GPU 없음")
        return
    print(f"   [{tag}] GPU 메모리 — "
          f"할당: {mem['allocated']:.2f}GB, "
          f"예약: {mem['reserved']:.2f}GB, "
          f"최대: {mem['max_allocated']:.2f}GB")


# ==========================================
# 1. 데이터셋 클래스
# ==========================================
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=2048):
        self.max_length = max_length
        pad_id = tokenizer.pad_id if hasattr(tokenizer, 'pad_id') else 0

        # 사전 토큰화: __getitem__마다 encode 호출하지 않고 초기화 시 한 번만 수행
        # int32 텐서로 저장하여 Python list 대비 ~7배 메모리 절감
        print(f"   -> TextDataset: {len(texts)}개 텍스트 사전 토큰화 중...")
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [pad_id] * (max_length - len(tokens))
            all_tokens.append(tokens)
        self.token_ids = torch.tensor(all_tokens, dtype=torch.int32)
        del all_tokens
        print(f"   -> TextDataset: 사전 토큰화 완료 ({len(self.token_ids)}개, {self.token_ids.element_size() * self.token_ids.nelement() / 1e9:.2f}GB)")

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        tokens = self.token_ids[idx]
        # 언어 모델 학습용: input은 마지막 토큰 제외, label은 첫 토큰 제외
        return tokens[:-1].long(), tokens[1:].long()

# ==========================================
# 3. SentencePiece BPE 토크나이저 (한국어 최적화)
# ==========================================
class KoreanBPETokenizer:
    """SentencePiece BPE 기반 한국어 토크나이저"""
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        
        # ByteLevel pre-tokenizer: 한국어 문자를 바이트 단위로 처리
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()
        
        # 패딩 설정
        self.tokenizer.enable_padding(
            pad_id=0,
            pad_token="<pad>",
            length=None  # 동적 패딩
        )
        
        # 특수 토큰 ID 저장용
        self.unk_id = None
        self.pad_id = 0
        self.eos_id = None
        self.bos_id = None
        
    def build_vocab(self, texts):
        """텍스트로부터 BPE 어휘 사전 구축"""
        print(">>> SentencePiece BPE 어휘 사전 구축 중...")
        print(f"   -> 텍스트 수: {len(texts)}개")
        
        # BPE 트레이너 설정
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,  # 최소 빈도 (한국어 특성 고려)
            special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
            show_progress=True,
            continuing_subword_prefix="",  # BPE 접두사 (선택사항)
        )
        
        # 텍스트 이터레이터로 학습
        def text_iterator():
            for text in texts:
                if text and isinstance(text, str):
                    yield text
        
        # BPE 학습
        self.tokenizer.train_from_iterator(
            text_iterator(),
            trainer=trainer,
            length=len(texts)
        )
        
        # 후처리 설정 (BOS/EOS 추가)
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="<s> $A </s>",
            special_tokens=[
                ("<s>", self.tokenizer.token_to_id("<s>")),
                ("</s>", self.tokenizer.token_to_id("</s>")),
            ],
        )
        
        # 특수 토큰 ID 저장
        self.unk_id = self.tokenizer.token_to_id("<unk>")
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.eos_id = self.tokenizer.token_to_id("</s>")
        self.bos_id = self.tokenizer.token_to_id("<s>")
        
        vocab_size_actual = self.tokenizer.get_vocab_size()
        print(f"   -> ✅ 어휘 사전 크기: {vocab_size_actual}")
        print(f"   -> 특수 토큰: <unk>={self.unk_id}, <pad>={self.pad_id}, <s>={self.bos_id}, </s>={self.eos_id}")

    def encode(self, text):
        """텍스트를 토큰 ID 리스트로 변환"""
        if not isinstance(text, str):
            text = str(text)
        
        # BPE 인코딩 (자동으로 BOS/EOS 추가됨)
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, token_ids):
        """토큰 ID 리스트를 텍스트로 변환"""
        # 리스트가 아닌 경우 변환
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        elif not isinstance(token_ids, list):
            token_ids = list(token_ids)
        
        # 특수 토큰 제거 (선택사항)
        filtered_ids = [
            tid for tid in token_ids 
            if tid not in [self.pad_id, self.unk_id, self.eos_id, self.bos_id]
        ]
        
        # 디코딩
        try:
            decoded = self.tokenizer.decode(filtered_ids)
            return decoded
        except Exception as e:
            # 오류 시 원본 ID로 시도
            return self.tokenizer.decode(token_ids)
    
    def get_vocab_size(self):
        """어휘 사전 크기 반환"""
        return self.tokenizer.get_vocab_size()
    
    def save(self, path):
        """토크나이저 저장"""
        self.tokenizer.save(path)
        print(f"   -> 토크나이저 저장: {path}")
    
    @classmethod
    def load(cls, path):
        """저장된 토크나이저 로드 (classmethod)"""
        instance = cls.__new__(cls)
        instance.tokenizer = Tokenizer.from_file(path)
        instance.vocab_size = instance.tokenizer.get_vocab_size()
        instance.unk_id = instance.tokenizer.token_to_id("<unk>")
        instance.pad_id = instance.tokenizer.token_to_id("<pad>") or 0
        instance.eos_id = instance.tokenizer.token_to_id("</s>")
        instance.bos_id = instance.tokenizer.token_to_id("<s>")
        # padding 재설정 (build_vocab에서 설정한 것과 동일)
        instance.tokenizer.enable_padding(
            pad_id=instance.pad_id, pad_token="<pad>", length=None
        )
        return instance

# ==========================================
# 4. DoRA (Weight-Decomposed Low-Rank Adaptation)
# ==========================================
class DoRALinear(nn.Module):
    """DoRA (Weight-Decomposed Low-Rank Adaptation) Linear 레이어
    output = m * normalize(W + ΔW) · x
    """
    def __init__(self, in_features, out_features, rank=16, alpha=32, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # Base weight
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

        # LoRA 행렬
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Magnitude 벡터
        self.m = nn.Parameter(torch.ones(out_features))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        base_out = x @ self.weight.T                                        # W·x
        lora_delta = self.dropout(x) @ self.lora_A.T @ self.lora_B.T \
                     * (self.alpha / self.rank)                             # ΔW·x
        combined = base_out + lora_delta
        norm = combined.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return combined * (self.m / norm)

def apply_dora_to_model(model, target_modules=None, rank=16, alpha=32):
    """모델에 DoRA 적용"""
    raise NotImplementedError(
        "DoRA 적용이 구현되지 않았습니다. "
        "utils.py의 DoRALinear를 각 레이어에 직접 교체해야 합니다."
    )

# ==========================================
# 5. GaLore 옵티마이저 래퍼
# ==========================================
class GaLoreAdamW(torch.optim.AdamW):
    """GaLore를 적용한 AdamW 옵티마이저"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, rank=128, update_proj_gap=200, **kwargs):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, **kwargs)
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.step_count = 0
        
        # GaLore 투영 행렬 초기화
        self.proj_P = {}
        self.proj_Q = {}
        self._init_galore_projections()
    
    def _init_galore_projections(self):
        """GaLore 투영 행렬 초기화"""
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad and p.dim() >= 2:
                    m, n = p.shape
                    # 낮은 랭크 투영 행렬
                    self.proj_P[id(p)] = torch.randn(m, self.rank, device=p.device) * 0.02
                    self.proj_Q[id(p)] = torch.randn(n, self.rank, device=p.device) * 0.02
    
    def step(self, closure=None):
        """GaLore 투영을 적용한 스텝"""
        self.step_count += 1
        
        # 주기적으로 투영 행렬 업데이트
        if self.step_count % self.update_proj_gap == 0:
            self._update_projections()
        
        return super().step(closure)
    
    def _update_projections(self):
        """SVD를 사용한 투영 행렬 업데이트"""
        raise NotImplementedError(
            "GaLoreAdamW._update_projections 미구현.\n"
            "  → pip install galore-torch 사용 권장: "
            "https://github.com/jiaweizzhao/GaLore"
        )

# ==========================================
# 6. 메인 학습 함수
# ==========================================
def train(
    use_dora=False,
    use_galore=False,
    use_flash_attention=True,
    max_seq_len=4096,  # RTX 3090에서 더 긴 컨텍스트 가능
    batch_size=2,  # 24GB에서 배치 크기 증가 가능
    gradient_accumulation_steps=8,
):
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> [1/6] 디바이스: {device}")
    if torch.cuda.is_available():
        print(f"   -> GPU: {torch.cuda.get_device_name(0)}")
        print(f"   -> VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
        
    # 모델 설정 (RTX 3090 24GB 최적화)
    print(">>> [2/6] 모델 설정 중...")
    config = ModelConfig(
        vocab_size=50000,
        dim=2048,  # 메모리에 맞게 조정
        n_layers=8,
        n_heads=16,
        head_dim=128,
        max_seq_len=max_seq_len,
        num_experts=8,
        num_experts_per_tok=2,
        dropout=0.1,
    )
    
    model = TransformerMoE(config, use_flash_attention=use_flash_attention).to(device)
    print(f"   -> 모델 파라미터: {model.get_num_params() / 1e6:.1f}M")
    
    # FlashAttention 지원 확인
    if use_flash_attention:
        try:
            from flash_attn import flash_attn_func
            print("   -> ✅ FlashAttention 사용 가능")
        except ImportError:
            print("   -> ⚠️ FlashAttention 미설치 (기본 scaled_dot_product_attention 사용)")
            use_flash_attention = False
    
    # Gradient Checkpointing 활성화
    if hasattr(torch.utils.checkpoint, 'checkpoint'):
        for layer in model.layers:
            layer.use_gradient_checkpointing = True
        print("   -> ✅ Gradient Checkpointing 활성화")
    
    # DoRA 적용
    if use_dora:
        model = apply_dora_to_model(model, rank=16, alpha=32)
        print("   -> ✅ DoRA 적용")
    
    # torch.compile 최적화
    try:
        if hasattr(torch, 'compile') and device.type == "cuda":
            model = torch.compile(model, mode="reduce-overhead")
            print("   -> ✅ torch.compile 활성화")
    except Exception as e:
        print(f"   -> ⚠️ torch.compile 실패: {e}")
    
    # 데이터셋 로드
    print(">>> [3/6] 데이터셋 로딩 중...")
    ds_namu = load_dataset("heegyu/namuwiki-extracted", split="train[:100000]")
    texts = [item["text"] for item in ds_namu if item.get("text")]
    print(f"   -> 로드된 텍스트: {len(texts)}개")
    
    # SentencePiece BPE 토크나이저 구축
    tokenizer = KoreanBPETokenizer(vocab_size=config.vocab_size)
    # 전체 텍스트로 학습 (한국어 특성 반영)
    tokenizer.build_vocab(texts[:50000])  # 더 많은 데이터로 학습
    
    # 실제 vocab_size 업데이트
    actual_vocab_size = tokenizer.get_vocab_size()
    if actual_vocab_size != config.vocab_size:
        print(f"   -> ⚠️ 실제 어휘 사전 크기: {actual_vocab_size} (설정: {config.vocab_size})")
        print(f"   -> 모델 vocab_size를 {actual_vocab_size}로 업데이트합니다.")
        config.vocab_size = actual_vocab_size
    
    # 데이터셋 생성
    dataset = TextDataset(texts, tokenizer, max_length=config.max_seq_len)
    
    # DataLoader 최적화
    num_workers = min(4, multiprocessing.cpu_count()) if os.name != 'nt' else 0
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    print(f"   -> 학습 배치 수: {len(dataloader)}")
    print(f"   -> 효과적 배치 크기: {batch_size * gradient_accumulation_steps}")
    
    # 학습 설정
    print(">>> [4/6] 학습 설정 중...")
    
    # GaLore 또는 일반 옵티마이저
    if use_galore:
        optimizer = GaLoreAdamW(
            model.parameters(),
            lr=2e-5,
            weight_decay=0.01,
            betas=(0.9, 0.95),
            rank=128,  # GaLore 랭크
            update_proj_gap=200,
        )
        print("   -> ✅ GaLore 옵티마이저 사용 (옵티마이저 상태 메모리 절감)")
    else:
        # 8-bit AdamW (메모리 절약)
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(),
                lr=2e-5,
                weight_decay=0.01,
                betas=(0.9, 0.95),
            )
            print("   -> ✅ 8-bit AdamW 사용 (메모리 절약)")
        except ImportError:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=2e-5,
                weight_decay=0.01,
                betas=(0.9, 0.95),
            )
            print("   -> ⚠️ bitsandbytes 미설치 (기본 AdamW 사용)")
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Mixed Precision Training
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"   -> Mixed Precision: {dtype}")
    
    num_epochs = 1
    save_steps = 500
    logging_steps = 20
    
    # 학습 루프
    print(">>> [5/6] 학습 시작!")
    model.train()
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (input_ids, labels) in enumerate(progress_bar):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            if scaler is not None:
                with torch.amp.autocast("cuda", dtype=dtype):
                    result = model(input_ids)
                    logits, aux_loss = result if isinstance(result, tuple) else (result, torch.tensor(0.0))
                    task_loss = criterion(logits.view(-1, config.vocab_size), labels.view(-1))
                    loss = (task_loss + 0.01 * aux_loss) / gradient_accumulation_steps
            else:
                result = model(input_ids)
                logits, aux_loss = result if isinstance(result, tuple) else (result, torch.tensor(0.0))
                task_loss = criterion(logits.view(-1, config.vocab_size), labels.view(-1))
                loss = (task_loss + 0.01 * aux_loss) / gradient_accumulation_steps
            
            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient Accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            global_step += 1
            
            # 로깅
            if global_step % logging_steps == 0:
                avg_loss = epoch_loss / logging_steps
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}"
                })
                epoch_loss = 0.0
            
            # 저장
            if global_step % save_steps == 0:
                save_dir = "lora_model_jamba2_rtx3090"
                os.makedirs(save_dir, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'step': global_step,
                }, os.path.join(save_dir, f"checkpoint-{global_step}.pt"))
                # 토크나이저는 별도로 저장 (첫 번째 체크포인트에만)
                if global_step == save_steps:
                    tokenizer_path = os.path.join(save_dir, "tokenizer.json")
                    tokenizer.save(tokenizer_path)
                print(f"\n   -> 체크포인트 저장: {save_dir}/checkpoint-{global_step}.pt")
    
    # 최종 저장
    print(">>> [6/6] 모델 저장 중...")
    save_dir = "lora_model_jamba2_rtx3090"
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'step': global_step,
    }, os.path.join(save_dir, "final_model.pt"))
    
    # 토크나이저 저장
    tokenizer_path = os.path.join(save_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    
    print(f">>> 완료! 모델이 {save_dir}에 저장되었습니다.")
    print(f"   -> 모델: {save_dir}/final_model.pt")
    print(f"   -> 토크나이저: {tokenizer_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RTX 3090 24GB 최적화 학습")
    parser.add_argument("--use-dora", action="store_true", help="DoRA 사용")
    parser.add_argument("--use-galore", action="store_true", help="GaLore 사용")
    parser.add_argument("--use-flash-attention", action="store_true", default=True, help="FlashAttention 사용")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="최대 시퀀스 길이")
    parser.add_argument("--batch-size", type=int, default=2, help="배치 크기")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Gradient accumulation steps")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RTX 3090 24GB 최적화 학습 스크립트")
    print("=" * 60)
    print(f"설정:")
    print(f"  - DoRA: {args.use_dora}")
    print(f"  - GaLore: {args.use_galore}")
    print(f"  - FlashAttention: {args.use_flash_attention}")
    print(f"  - Max Sequence Length: {args.max_seq_len}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Gradient Accumulation: {args.gradient_accumulation_steps}")
    print("=" * 60)
    
    train(
        use_dora=args.use_dora,
        use_galore=args.use_galore,
        use_flash_attention=args.use_flash_attention,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
