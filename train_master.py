"""
Master 모델 학습 스크립트
1단계: Master 모델 (MoE 없이 기본 FFN만) 학습
2단계: (추후) 라우터로 MoE 추가
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from datasets import load_dataset

from model import ModelConfig, MasterBlock, precompute_freqs_cis, RMSNorm
# 하위 호환용 별칭
JambaConfig = ModelConfig
from utils import TextDataset, KoreanBPETokenizer, get_gpu_memory, print_memory_stats

# ==========================================
# 1. Master 모델 클래스 (MoE 없이 FFN만 사용)
# ==========================================
class MasterModel(nn.Module):
    """Master 모델: MoE 없이 기본 FFN만 사용하는 Jamba2 모델"""
    def __init__(self, config: ModelConfig, use_flash_attention=False):
        super().__init__()
        self.config = config
        
        # 임베딩
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        
        # 레이어 (MasterBlock 사용 - MoE 없이 FFN만)
        self.layers = nn.ModuleList([
            MasterBlock(config, use_flash_attention=use_flash_attention) 
            for _ in range(config.n_layers)
        ])
        
        # 출력
        self.norm = RMSNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Embedding weight tying (파라미터 공유로 메모리 절감)
        self.output.weight = self.token_embedding.weight

        # RoPE 주파수 사전 계산 (register_buffer로 자동 device 연동)
        self.register_buffer(
            'freqs_cis',
            precompute_freqs_cis(config.head_dim, config.max_seq_len * 2),
            persistent=False,
        )

        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, tokens=None, input_ids=None, **kwargs):
        # PEFT 호환성: input_ids 또는 tokens 모두 지원
        if input_ids is not None:
            tokens = input_ids
        elif tokens is None:
            raise ValueError("Either 'tokens' or 'input_ids' must be provided")
        
        bsz, seqlen = tokens.shape
        
        # 임베딩
        h = self.token_embedding(tokens)
        
        # RoPE 주파수 (register_buffer이므로 이미 올바른 device)
        freqs_cis = self.freqs_cis[:seqlen]
        
        # 레이어 통과
        for layer in self.layers:
            h, _ = layer(h, freqs_cis)

        # 출력
        h = self.norm(h)
        output = self.output(h)

        return output
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """PEFT 호환성을 위한 메서드"""
        return {"input_ids": input_ids}
    
    def get_input_embeddings(self):
        """PEFT 호환성을 위한 메서드"""
        return self.token_embedding
    
    def set_input_embeddings(self, value):
        """PEFT 호환성을 위한 메서드"""
        self.token_embedding = value
    
    def get_output_embeddings(self):
        """PEFT 호환성을 위한 메서드"""
        return self.output
    
    def set_output_embeddings(self, value):
        """PEFT 호환성을 위한 메서드"""
        self.output = value
    
    def get_num_params(self):
        """모델 파라미터 수 계산"""
        return sum(p.numel() for p in self.parameters())
    
    def print_param_stats(self):
        """모델 파라미터 통계 출력"""
        total_params = 0
        print("\n>>> 모델 파라미터 통계:")
        
        # 임베딩
        emb_params = sum(p.numel() for p in self.token_embedding.parameters())
        total_params += emb_params
        print(f"   📝 임베딩: {emb_params / 1e6:.2f}M ({emb_params:,})")
        
        # 레이어별 통계
        layer_total = 0
        for layer_idx, layer in enumerate(self.layers):
            # Attention
            attn_params = sum(p.numel() for p in layer.attention.parameters())
            attn_norm_params = sum(p.numel() for p in layer.attention_norm.parameters())
            
            # FFN (Master 모델은 단일 FFN)
            ffn_params = sum(p.numel() for p in layer.ffn.parameters())
            ffn_norm_params = sum(p.numel() for p in layer.ffn_norm.parameters())
            
            layer_params = attn_params + attn_norm_params + ffn_params + ffn_norm_params
            layer_total += layer_params
            
            if layer_idx < 2 or layer_idx == len(self.layers) - 1:  # 처음 2개와 마지막만 상세 출력
                print(f"   🔹 Layer {layer_idx}:")
                print(f"      - Attention: {attn_params / 1e6:.2f}M")
                print(f"      - Attention Norm: {attn_norm_params / 1e3:.1f}K")
                print(f"      - FFN: {ffn_params / 1e6:.2f}M")
                print(f"      - FFN Norm: {ffn_norm_params / 1e3:.1f}K")
                print(f"      - Total: {layer_params / 1e6:.2f}M")
        
        if len(self.layers) > 3:
            print(f"   ... ({len(self.layers) - 3}개 레이어 생략)")
        
        total_params += layer_total
        print(f"   📊 레이어 총합 ({len(self.layers)}개): {layer_total / 1e6:.2f}M")
        
        # 출력 레이어
        norm_params = sum(p.numel() for p in self.norm.parameters())
        output_params = sum(p.numel() for p in self.output.parameters())
        total_params += norm_params + output_params
        print(f"   📤 출력 레이어:")
        print(f"      - Norm: {norm_params / 1e3:.1f}K")
        print(f"      - Output: {output_params / 1e6:.2f}M")
        
        print(f"\n   ✅ 총 파라미터: {total_params / 1e6:.2f}M ({total_params:,})")
        return total_params

# ==========================================
# 2. Master 모델 학습 함수
# ==========================================
def train_master_model(
    model, 
    dataloader, 
    optimizer, 
    criterion,
    device,
    num_epochs: int = 1,
    save_dir: str = "master_model_checkpoints",
    use_mixed_precision: bool = True,
    save_every_n_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
):
    """Master 모델 학습 실행"""
    print(f"\n>>> Master 모델 학습 시작!")
    print(f"   -> Epochs: {num_epochs}")
    
    model.train()
    # 새로운 API 사용 (FutureWarning 해결)
    scaler = torch.amp.GradScaler('cuda') if use_mixed_precision and device.type == "cuda" else None
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    os.makedirs(save_dir, exist_ok=True)
    all_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Master Model, Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (input_ids, labels) in enumerate(progress_bar):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            if scaler is not None:
                # 새로운 API 사용 (FutureWarning 해결)
                with torch.amp.autocast('cuda', dtype=dtype):
                    logits = model(input_ids)
                    loss = criterion(logits.view(-1, model.config.vocab_size), labels.view(-1))
                    # Gradient accumulation을 위한 loss 정규화
                    loss = loss / gradient_accumulation_steps
            else:
                logits = model(input_ids)
                loss = criterion(logits.view(-1, model.config.vocab_size), labels.view(-1))
                loss = loss / gradient_accumulation_steps
            
            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation: 지정된 스텝마다 업데이트
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
            
            epoch_loss += loss.item() * gradient_accumulation_steps  # 원래 loss로 복원
            
            # 메모리 정리는 OOM 방지용으로 드물게 실행 (빈번한 호출은 오히려 성능 저하)
            if batch_idx % 1000 == 0 and batch_idx > 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # 진행 상황 업데이트
            if batch_idx % 20 == 0:
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{epoch_loss/(batch_idx+1):.4f}",
                    "mem": f"{get_gpu_memory()['allocated']:.1f}GB" if get_gpu_memory() else "N/A"
                })
        
        # 잔여 gradient 처리 (마지막 배치가 accumulation 배수가 아닌 경우)
        if len(dataloader) % gradient_accumulation_steps != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        avg_epoch_loss = epoch_loss / len(dataloader)
        all_losses.append(avg_epoch_loss)
        print(f"   -> Epoch {epoch+1} 완료. 평균 Loss: {avg_epoch_loss:.4f}")
        
        # 주기적으로 체크포인트 저장
        if (epoch + 1) % save_every_n_epochs == 0:
            checkpoint_path = os.path.join(save_dir, f"master_model_epoch_{epoch+1}.pt")
            if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                config_to_save = model.base_model.model.config
            else:
                config_to_save = model.config
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config_to_save,
                'epoch': epoch + 1,
                'loss': avg_epoch_loss,
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"   -> 체크포인트 저장: {checkpoint_path}")
    
    return all_losses


def build_qlora_master_model(config, device='cpu'):
    """QLoRA 적용된 Master 모델 생성 (체크포인트 로드 후 merge용)."""
    from peft import LoraConfig, get_peft_model
    import bitsandbytes as bnb

    model = MasterModel(config, use_flash_attention=False)
    compute_dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16

    for layer in model.layers:
        for name in ['wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w3']:
            parent = layer.attention if name in ('wq','wk','wv','wo') else layer.ffn
            old_linear = getattr(parent, name)
            new_linear = bnb.nn.Linear4bit(
                old_linear.in_features, old_linear.out_features,
                bias=old_linear.bias is not None,
                compute_dtype=compute_dtype, quant_type='nf4'
            )
            with torch.no_grad():
                # clone() 대신 직접 대입 후 원본 해제 (피크 메모리 절감)
                new_linear.weight.data = old_linear.weight.data
                if old_linear.bias is not None:
                    new_linear.bias.data = old_linear.bias.data
            setattr(parent, name, new_linear)
            del old_linear

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["wq", "wk", "wv", "wo", "w1", "w2", "w3"],
        lora_dropout=0.05, bias="none", task_type="FEATURE_EXTRACTION",
    )
    model = get_peft_model(model, lora_config)
    return model.to(device)


def _dequantize_4bit_weight(weight_param, base_layer, device='cpu'):
    """Linear4bit 가중치를 full-precision [out_features, in_features]로 디양자화."""
    import bitsandbytes as bnb
    out_features = base_layer.out_features
    in_features = base_layer.in_features
    expected_shape = (out_features, in_features)
    quant_state = getattr(weight_param, "quant_state", None)
    if quant_state is None:
        # 아직 양자화되지 않은 경우 (CPU 로드 직후)
        w = weight_param.data.clone().to(device=device, dtype=torch.float32)
    else:
        # bnb.functional.dequantize_4bit 사용 (weight.dequantize()는 shape 이슈 가능)
        w = bnb.functional.dequantize_4bit(
            weight_param.data, quant_state=quant_state
        ).to(device=device, dtype=torch.float32)
    # bitsandbytes 버전에 따라 shape가 [in,out] 또는 flatten될 수 있음 → 명시적 reshape
    if w.shape != expected_shape:
        w = w.reshape(expected_shape)
    return w


def _qlora_to_full_state_dict(peft_model, device='cpu'):
    """
    QLoRA(PEFT) 모델에서 merge_and_unload() 없이 직접 base 4bit + LoRA를 합쳐
    full-precision state_dict 추출. (PEFT merge_and_unload는 4bit base와 shape 불일치로 실패할 수 있음)
    """
    import bitsandbytes as bnb
    base = peft_model.base_model.model
    state = {}
    # LoRA scaling: alpha / r (PEFT 기본). peft_config는 adapter 이름 -> config
    scaling = 2.0
    if hasattr(peft_model, "peft_config"):
        cfg = peft_model.peft_config.get("default", None) if isinstance(peft_model.peft_config, dict) else peft_model.peft_config
        if cfg is not None:
            a, r = getattr(cfg, "lora_alpha", 32), getattr(cfg, "r", 16)
            scaling = float(a) / float(r)

    for name, mod in base.named_modules():
        base_layer = getattr(mod, "base_layer", mod)
        lora_A = getattr(mod, "lora_A", None)
        lora_B = getattr(mod, "lora_B", None)
        is_lora = lora_A is not None and lora_B is not None
        if list(mod.children()) and not is_lora:
            continue  # LoRA가 아닌 부모 모듈은 건너뜀
        prefix = (name + '.') if name else ''
        if is_lora:
            # LoRA 적용 레이어: base(4bit) + scaling * (B @ A)
            if isinstance(base_layer, bnb.nn.Linear4bit):
                w = _dequantize_4bit_weight(base_layer.weight, base_layer, device=device)
            else:
                w = base_layer.weight.data.clone().to(device=device, dtype=torch.float32)
            # ModuleDict에서 adapter 모듈 추출 (키: "default" 또는 첫 번째)
            def _get_lora_weight(m):
                if hasattr(m, "get") and m.get("default", None) is not None:
                    m = m["default"]
                elif hasattr(m, "values") and len(m) > 0:
                    m = next(iter(m.values()))
                return m.weight if hasattr(m, "weight") else m
            A = _get_lora_weight(lora_A)
            B = _get_lora_weight(lora_B)
            A = A.to(device=device, dtype=torch.float32)
            B = B.to(device=device, dtype=torch.float32)
            # PEFT LoRA: base w는 [out,in]. delta는 [out,in]이어야 함.
            # A [r,in], B [out,r] → B @ A = [out,in].  A [in,r], B [r,out] → (A @ B).T = [out,in]
            out_f, in_f = w.shape[0], w.shape[1]
            if B.shape[0] == out_f and A.shape[1] == in_f:
                delta = (B @ A) * scaling
            elif A.shape[0] == in_f and B.shape[1] == out_f:
                delta = (A @ B).T * scaling
            else:
                raise RuntimeError(
                    f"LoRA delta shape mismatch at {name}: w{w.shape}, A{A.shape}, B{B.shape}"
                )
            if delta.shape != w.shape:
                delta = delta.reshape(w.shape)
            w = w + delta
            state[prefix + 'weight'] = w
            if getattr(base_layer, "bias", None) is not None:
                state[prefix + 'bias'] = base_layer.bias.data.to(device=device)
        elif isinstance(mod, bnb.nn.Linear4bit):
            w = _dequantize_4bit_weight(mod.weight, mod, device=device)
            state[prefix + 'weight'] = w
            if mod.bias is not None:
                state[prefix + 'bias'] = mod.bias.data.to(device=device)
        else:
            for k, v in mod.state_dict().items():
                state[prefix + k] = v.to(device) if hasattr(v, 'to') else v
    return state


def load_master_for_moe(checkpoint_path, device='cpu'):
    """
    Master 체크포인트를 MoE 단계용으로 로드.
    - 일반 체크포인트: 그대로 load_state_dict.
    - QLoRA 체크포인트 (base_model.model.* 키): QLoRA 구조로 로드 후 merge + 디양자화하여
      full-precision MasterModel 인스턴스를 반환.
    """
    if isinstance(device, str):
        device = torch.device(device)
    import warnings
    warnings.warn(
        f"weights_only=False로 {checkpoint_path}를 로드합니다. "
        "신뢰할 수 없는 파일이라면 보안 위험이 있습니다.",
        UserWarning,
        stacklevel=2,
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    state = ckpt['model_state_dict']
    first_key = next(iter(state.keys()))

    if 'base_model.model.' in first_key:
        # QLoRA 형식: 동일 구조로 복원 → merge_and_unload() 없이 직접 base+LoRA 합쳐 full-precision 추출
        try:
            qlora = build_qlora_master_model(config, device=device)
            qlora.load_state_dict(state, strict=False)
            full_state = _qlora_to_full_state_dict(qlora, device=device)
            del qlora
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            raise RuntimeError(
                f"QLoRA 체크포인트를 full-precision으로 변환하는 중 오류: {e}\n"
                "peft, bitsandbytes가 설치되어 있는지 확인하세요."
            ) from e
        master = MasterModel(config, use_flash_attention=False)
        master.load_state_dict(full_state, strict=True)
        return master

    master = MasterModel(config, use_flash_attention=False)
    master.load_state_dict(state, strict=True)
    return master


def main():
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 디바이스: {device}")
    if torch.cuda.is_available():
        print(f"   -> GPU: {torch.cuda.get_device_name(0)}")
        print(f"   -> VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # QLoRA 사용 여부 설정 (메모리 계산 전에 설정)
    use_qlora = True  # QLoRA 사용 시 메모리 대폭 절약
    
    # 설정 (QLoRA 사용 시 최적화 - 메모리 여유로 더 큰 모델 가능)
    # QLoRA 사용 시: 4-bit 모델(~0.5GB per 1B) + LoRA(~수십MB) + optimizer(~LoRA만) + activation ≈ 5-10GB
    # Full fine-tuning 대비: ~70-80% 메모리 절약
    if use_qlora:
        # QLoRA: FP32 모델 생성(RAM) → 4-bit 양자화(VRAM) 순서
        # dim=2560, n_layers=32 → ~2.6B params, FP32 생성 시 ~10.5GB RAM, 4-bit VRAM ~1.3GB
        # + LoRA(~수십MB) + optimizer(~LoRA만) + activation ≈ 총 ~8-12GB VRAM
        config = ModelConfig(
            vocab_size=32000,
            dim=2560,           # ~2.6B 파라미터 (FP32 ~10.5GB RAM, 4-bit ~1.3GB VRAM)
            n_layers=32,
            n_heads=20,         # dim / head_dim = 2560 / 128 = 20
            head_dim=128,
            max_seq_len=2048,
            num_experts=1,      # MoE 사용 안 함 (master 모델)
            num_experts_per_tok=1,
            dropout=0.1,
        )
        print(">>> Master 모델 설정 (QLoRA 최적화 - RTX 3090 + 39GB RAM):")
        print(f"   -> 모델 차원: {config.dim}")
        print(f"   -> 레이어 수: {config.n_layers}")
        print(f"   -> 어텐션 헤드: {config.n_heads}")
        print(f"   -> 최대 시퀀스 길이: {config.max_seq_len}")
        print(f"   -> 예상 파라미터: ~2.6B")
        print(f"   -> FP32 생성 시 RAM: ~10.5GB, 4-bit VRAM: ~1.3GB + activation")
    else:
        # Full fine-tuning: 기존 설정 유지
        config = ModelConfig(
            vocab_size=32000,
            dim=2304,           # 24GB 초과 방지
            n_layers=26,
            n_heads=18,
            head_dim=128,
            max_seq_len=1920,
            num_experts=1,
            num_experts_per_tok=1,
            dropout=0.1,
        )
        print(">>> Master 모델 설정 (Full fine-tuning - 23GB VRAM 최적화):")
        print(f"   -> 모델 차원: {config.dim}")
        print(f"   -> 레이어 수: {config.n_layers}")
        print(f"   -> 어텐션 헤드: {config.n_heads}")
        print(f"   -> 예상 파라미터: ~1.4-1.8B")
        print(f"   -> 예상 메모리: ~20-23GB")
    
    print(f"   -> 헤드 차원: {config.head_dim}")
    print(f"   -> Vocab 크기: {config.vocab_size}")
    print(f"   -> MoE 사용: ❌ (FFN만 사용)")
    
    # ==========================================
    # QLoRA vs FlashAttention 선택
    # ==========================================
    # QLoRA: 메모리 절약 (70-80%), 더 큰 모델 학습 가능, LoRA만 학습 (성능 약간 저하 가능)
    # FlashAttention: 빠른 학습, Full fine-tuning, 메모리 절약 (activation만)
    # 
    # 추천:
    # - 24GB VRAM에서 더 큰 모델 학습: QLoRA 권장
    # - 작은 모델 + 빠른 학습: FlashAttention 권장
    # - 최대 성능: 둘 다 사용 불가 (QLoRA 사용 시 FlashAttention 비활성화)
    
    # QLoRA 사용 시 FlashAttention 비활성화 (4-bit와 호환되지 않음)
    use_flash_attention = not use_qlora  # QLoRA 사용 시 False
    
    # 배치 크기 설정 (QLoRA 사용 시 더 큰 배치 가능)
    if use_qlora:
        batch_size = 4  # QLoRA 사용 시 메모리 여유로 배치 크기 증가 (필요시 6-8로 증가 가능)
        gradient_accumulation_steps = 2  # 실제 배치 크기 = 8
        print(f"   -> 배치 크기: {batch_size} (QLoRA 사용 시 여유 있음, 필요시 더 증가 가능)")
    else:
        batch_size = 2  # Full fine-tuning: 작은 배치
        gradient_accumulation_steps = 2  # 실제 배치 크기 = 4
    
    print("\n>>> 학습 방법 선택:")
    if use_qlora:
        print("   ✅ QLoRA 선택 (4-bit 양자화 + LoRA)")
        print("      -> 장점: 메모리 70-80% 절약, 더 큰 모델 학습 가능, 더 큰 배치 크기")
        print("      -> 단점: FlashAttention 사용 불가, LoRA만 학습 (성능 약간 저하 가능)")
        print("      -> 예상 메모리: ~5-8GB (Full fine-tuning 대비)")
    else:
        print("   ✅ FlashAttention 선택 (Full fine-tuning)")
        print("      -> 장점: 빠른 학습 속도, 모든 파라미터 학습, 최대 성능")
        print("      -> 단점: 더 많은 메모리 필요, 작은 배치 크기")
        print("      -> 예상 메모리: ~20-23GB")
    
    # 모델 생성
    print("\n>>> Master 모델 생성 중...")
    if use_qlora:
        print("   -> QLoRA 사용: FlashAttention 자동 비활성화 (4-bit 양자화와 호환되지 않음)")
        # 주의: build_qlora_master_model 내부에서 MasterModel을 생성하므로 여기서 중복 생성하지 않음
    else:
        model = MasterModel(config, use_flash_attention=use_flash_attention)

    # QLoRA 적용 (4-bit 양자화 + LoRA)
    if use_qlora:
        try:
            print("   -> QLoRA 적용 중 (4-bit 양자화 + LoRA)...")
            model = build_qlora_master_model(config, device=device)

            # 학습 가능한 파라미터 확인
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   -> ✅ QLoRA 적용 완료")
            print(f"   -> 학습 가능한 파라미터: {trainable_params / 1e6:.2f}M / {total_params / 1e6:.2f}M ({100 * trainable_params / total_params:.2f}%)")

        except (ImportError, AttributeError, Exception) as e:
            print(f"   -> ⚠️ QLoRA 적용 실패: {e}")
            import traceback
            traceback.print_exc()
            print("   -> pip install peft bitsandbytes 설치 필요")
            print("   -> QLoRA 없이 학습 진행...")
            use_qlora = False
            model = model.to(device)
            model.print_param_stats()
    else:
        model = model.to(device)
        model.print_param_stats()
    
    # 메모리 사용량 확인
    total_estimated = 0.0
    if torch.cuda.is_available():
        if use_qlora:
            # QLoRA 사용 시: 4-bit 모델 + LoRA 어댑터만 학습
            # 4-bit 모델은 약 0.5GB per 1B 파라미터
            total_params = sum(p.numel() for p in model.parameters())
            model_memory = (total_params * 0.5) / 1e9  # 4-bit 양자화
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            lora_memory = (trainable_params * 2) / 1e9  # LoRA 파라미터 (fp16)
            model_memory_total = model_memory + lora_memory
            
            # QLoRA는 LoRA 파라미터만 학습하므로 optimizer 메모리도 크게 절약
            optimizer_memory = lora_memory * 2  # LoRA 파라미터에 대한 optimizer만
            # Gradient checkpointing 사용 시 activation 메모리 절약
            activation_memory = (batch_size * config.max_seq_len * config.dim * config.n_layers * 1.5) / 1e9
            total_estimated = model_memory_total + optimizer_memory + activation_memory
            
            print(f"   -> 모델 메모리 (4-bit): {model_memory:.2f}GB")
            print(f"   -> LoRA 어댑터: {lora_memory:.2f}GB")
            print(f"   -> 옵티마이저 (LoRA만): {optimizer_memory:.2f}GB")
            print(f"   -> 활성화: {activation_memory:.2f}GB")
            print(f"   -> 예상 총 메모리: {total_estimated:.2f}GB")
            print(f"   -> 💡 QLoRA 사용으로 메모리 약 70-80% 절약!")
            print(f"   -> ✅ 메모리 여유로 더 큰 모델 학습 가능!")
        else:
            model_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
            print(f"   -> 모델 메모리 (파라미터만): {model_memory:.2f}GB")
            
            # 8-bit optimizer 사용 시 예상 메모리
            optimizer_memory = model_memory * 1.5  # 8-bit optimizer는 약 1.5배
            # Activation 메모리 (대략적 추정)
            # Gradient checkpointing 사용 시 activation 메모리 절약
            activation_memory = (batch_size * config.max_seq_len * config.dim * config.n_layers * 1.5) / 1e9  # forward + backward (checkpointing으로 절약)
            total_estimated = model_memory + optimizer_memory + activation_memory
            print(f"   -> 옵티마이저: {optimizer_memory:.2f}GB")
            print(f"   -> 활성화: {activation_memory:.2f}GB")
            print(f"   -> 예상 총 메모리: {total_estimated:.2f}GB")
        
        if total_estimated > 24.0:
            print(f"   -> ❌ 경고: 예상 메모리가 24GB를 초과합니다! ({total_estimated:.2f}GB)")
            print(f"   -> 배치 크기를 1로 줄이거나 모델 크기를 더 줄이세요")
        elif total_estimated > 23.0:
            print(f"   -> ⚠️ 경고: 예상 메모리가 23GB를 초과합니다! ({total_estimated:.2f}GB)")
            print(f"   -> 배치 크기를 1로 줄이거나 모델 크기를 조정하세요")
        elif total_estimated > 21.0:
            print(f"   -> ⚠️ 주의: 예상 메모리가 21GB를 초과합니다. ({total_estimated:.2f}GB)")
            print(f"   -> Gradient checkpointing이 활성화되어 있어 실제 사용량은 더 낮을 수 있습니다.")
        else:
            print(f"   -> ✅ 예상 메모리가 23GB 이하입니다. ({total_estimated:.2f}GB)")
    
    # 데이터셋 준비
    print("\n>>> 데이터셋 로딩 중...")
    try:
        ds_namu = load_dataset("heegyu/namuwiki-extracted", split="train[:10000]")
        texts = [item["text"] for item in ds_namu if item.get("text")]
        del ds_namu  # HuggingFace Dataset 메모리 해제
        import gc; gc.collect()
    except:
        # 데이터셋 로딩 실패시 더미 데이터
        print("   -> ⚠️ 데이터셋 로딩 실패, 더미 데이터 사용")
        texts = ["안녕하세요. 이것은 테스트 텍스트입니다."] * 1000

    print(f"   -> 텍스트 수: {len(texts)}")

    # 토크나이저
    tokenizer = KoreanBPETokenizer(vocab_size=config.vocab_size)
    tokenizer.build_vocab(texts[:2000])
    
    # 메모리 사용량에 따라 배치 크기 동적 조정
    if torch.cuda.is_available() and total_estimated > 23.0:
        batch_size = 1  # 메모리가 부족하면 배치 크기 1로 감소
        print(f"   -> ⚠️ 예상 메모리가 23GB 초과로 배치 크기를 1로 조정합니다")
    
    # 데이터로더 (23GB VRAM에 맞게 배치 크기 조정)
    dataset = TextDataset(texts, tokenizer, max_length=config.max_seq_len)
    # 사전 토큰화 완료 → __getitem__이 텐서 슬라이싱만 하므로 num_workers=0으로 충분
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"   -> 배치 크기: {batch_size} (메모리 부족 시 자동으로 1로 조정)")
    print(f"   -> Gradient Accumulation Steps: {gradient_accumulation_steps} (실제 배치 크기: {batch_size * gradient_accumulation_steps})")
    
    print(f"   -> 배치 수: {len(dataloader)}")
    
    # 학습 설정
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # 옵티마이저 설정
    if use_qlora:
        # QLoRA 사용 시: LoRA 파라미터만 학습하므로 일반 optimizer 사용 가능
        # 하지만 8-bit optimizer를 사용하면 더 절약 가능
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(),
                lr=2e-4,  # LoRA는 보통 더 작은 학습률 사용
                weight_decay=0.01,
                betas=(0.9, 0.95),
            )
            print("   -> ✅ 8-bit AdamW 사용 (QLoRA + 메모리 절약)")
        except ImportError:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=2e-4,
                weight_decay=0.01,
                betas=(0.9, 0.95),
            )
            print("   -> ✅ AdamW 사용 (QLoRA)")
    else:
        # Full fine-tuning: 8-bit 옵티마이저 사용 (메모리 절약)
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(),
                lr=3e-4,
                weight_decay=0.01,
                betas=(0.9, 0.95),
            )
            print("   -> ✅ 8-bit AdamW 사용 (메모리 절약)")
        except ImportError:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=3e-4,
                weight_decay=0.01,
                betas=(0.9, 0.95),
            )
            print("   -> ⚠️ bitsandbytes 미설치 (기본 AdamW 사용, 메모리 사용량 증가)")
    
    # Gradient Checkpointing 강제 활성화 (메모리 절약)
    # PEFT 래핑 후에는 model.base_model.model.layers로 접근
    base = model
    if hasattr(model, 'base_model'):
        base = model.base_model
    if hasattr(base, 'model'):
        base = base.model
    layers = getattr(base, 'layers', [])
    ckpt_count = 0
    for layer in layers:
        if hasattr(layer, 'use_gradient_checkpointing'):
            layer.use_gradient_checkpointing = True
            ckpt_count += 1
    if ckpt_count > 0:
        print(f"   -> ✅ Gradient Checkpointing 활성화 ({ckpt_count}개 레이어, 메모리 절약)")
    
    # ==========================================
    # Master 모델 학습
    # ==========================================
    print("\n" + "="*60)
    print("MASTER 모델 학습 시작")
    print("="*60)
    
    losses = train_master_model(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=1,  # 원하는 epoch 수로 변경
        save_every_n_epochs=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    
    print_memory_stats("Master 모델 학습 완료")
    
    # ==========================================
    # 최종 결과 및 저장
    # ==========================================
    print("\n" + "="*60)
    print("Master 모델 학습 완료!")
    print("="*60)
    model.print_param_stats()
    print(f"✅ 평균 Loss: {sum(losses) / len(losses):.4f}")
    
    # 최종 모델 저장
    final_save_dir = "master_model_checkpoints"
    os.makedirs(final_save_dir, exist_ok=True)
    
    if use_qlora:
        # QLoRA 사용 시: LoRA 어댑터만 저장 (원본 모델은 양자화된 상태로 유지)
        try:
            from peft import PeftModel
            model.save_pretrained(final_save_dir)
            print(f"   -> LoRA 어댑터 저장: {final_save_dir}/adapter_model.bin")
            
            # 추가 정보 저장
            torch.save({
                'config': config,
                'losses': losses,
                'final_loss': losses[-1] if losses else None,
                'use_qlora': True,
            }, os.path.join(final_save_dir, "training_info.pt"))
            # MoE 단계용 full-precision 체크포인트 저장 (merge+디양자화, 로드 시 shape 이슈 회피)
            try:
                full_state = _qlora_to_full_state_dict(model, device='cpu')
                moe_path = os.path.join(final_save_dir, "master_model_for_moe.pt")
                torch.save({'model_state_dict': full_state, 'config': config}, moe_path)
                print(f"   -> MoE용 체크포인트 저장: {moe_path}")
            except Exception as e:
                print(f"   -> ⚠️ MoE용 체크포인트 저장 실패: {e}")
        except:
            # PEFT 저장 실패 시 일반 저장
            final_path = os.path.join(final_save_dir, "master_model_final.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'losses': losses,
                'final_loss': losses[-1] if losses else None,
                'use_qlora': True,
            }, final_path)
    else:
        final_path = os.path.join(final_save_dir, "master_model_final.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'losses': losses,
            'final_loss': losses[-1] if losses else None,
            'use_qlora': False,
        }, final_path)
    
    # 토크나이저 저장
    tokenizer.save(os.path.join(final_save_dir, "tokenizer.json"))
    
    print(f"\n>>> 최종 모델 저장: {final_save_dir}/")
    print(f"   -> master_model_final.pt (또는 adapter)")
    if use_qlora and os.path.exists(os.path.join(final_save_dir, "master_model_for_moe.pt")):
        print(f"   -> master_model_for_moe.pt (MoE용 full-precision)")
    print(f"   -> tokenizer.json")
    print(f"\n💡 다음 단계: train_moe.py로 MoE 확장 (master_model_for_moe.pt 우선 로드)")

if __name__ == "__main__":
    main()