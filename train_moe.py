"""
MoE 단계별 학습 스크립트
Master 모델 학습 후 → Router 학습 → Expert 학습 → 전체 미세조정
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import sys
import gc
from tqdm import tqdm
from datasets import load_dataset, interleave_datasets
import random

from model import ModelConfig, MoEBlock, precompute_freqs_cis, RMSNorm, convert_legacy_state_dict
from train_master import MasterModel, load_master_for_moe
from utils import TextDataset, KoreanBPETokenizer, get_gpu_memory, print_memory_stats

# ==========================================
# 1. MoE 모델 클래스
# ==========================================
class MoEModel(nn.Module):
    """MoE 모델: Master 모델을 MoE로 변환"""
    def __init__(self, config: ModelConfig, use_flash_attention=False):
        super().__init__()
        self.config = config
        
        # 임베딩
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        
        # 레이어 (MoEBlock: Attention + Router + Experts, Mixtral 방식)
        self.layers = nn.ModuleList([
            MoEBlock(config, use_flash_attention=use_flash_attention)
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
        
        # 레이어 통과 (aux_loss 수집)
        total_aux_loss = torch.tensor(0.0, device=h.device)
        for layer in self.layers:
            h, aux_loss = layer(h, freqs_cis)
            total_aux_loss = total_aux_loss + aux_loss

        # 출력
        h = self.norm(h)
        output = self.output(h)

        return output, total_aux_loss
    
    def freeze_experts(self, expert_indices=None):
        """Expert 파라미터 동결 (라우터만 학습)"""
        for layer in self.layers:
            moe = getattr(layer, 'moe', None)
            if moe is None or not hasattr(moe, 'experts'):
                continue
            if expert_indices is None:
                for expert in moe.experts:
                    for param in expert.parameters():
                        param.requires_grad = False
            else:
                for idx in expert_indices:
                    if idx < len(moe.experts):
                        for param in moe.experts[idx].parameters():
                            param.requires_grad = False

    def unfreeze_experts(self, expert_indices=None):
        """Expert 파라미터 해제"""
        for layer in self.layers:
            moe = getattr(layer, 'moe', None)
            if moe is None or not hasattr(moe, 'experts'):
                continue
            if expert_indices is None:
                for expert in moe.experts:
                    for param in expert.parameters():
                        param.requires_grad = True
            else:
                for idx in expert_indices:
                    if idx < len(moe.experts):
                        for param in moe.experts[idx].parameters():
                            param.requires_grad = True

    def freeze_routers(self):
        """Router 파라미터 동결 (Expert만 학습)"""
        for layer in self.layers:
            moe = getattr(layer, 'moe', None)
            if moe is not None and hasattr(moe, 'router'):
                for param in moe.router.parameters():
                    param.requires_grad = False

    def unfreeze_routers(self):
        """Router 파라미터 해제"""
        for layer in self.layers:
            moe = getattr(layer, 'moe', None)
            if moe is not None and hasattr(moe, 'router'):
                for param in moe.router.parameters():
                    param.requires_grad = True
    
    def freeze_attention(self):
        """Attention 파라미터 동결"""
        for layer in self.layers:
            for param in layer.attention.parameters():
                param.requires_grad = False
            for param in layer.attention_norm.parameters():
                param.requires_grad = False

    def unfreeze_attention(self):
        """Attention 파라미터 해제"""
        for layer in self.layers:
            for param in layer.attention.parameters():
                param.requires_grad = True
            for param in layer.attention_norm.parameters():
                param.requires_grad = True

    def get_num_params(self):
        """모델 파라미터 수 계산"""
        return sum(p.numel() for p in self.parameters())
    
    def print_param_stats(self):
        """모델 파라미터 통계 출력"""
        total_params = 0
        print("\n>>> MoE 모델 파라미터 통계:")

        # 임베딩
        emb_params = sum(p.numel() for p in self.token_embedding.parameters())
        total_params += emb_params
        print(f"   📝 임베딩: {emb_params / 1e6:.2f}M")

        # 레이어별 통계
        layer_total = 0
        router_total = 0
        expert_total = 0

        for layer_idx, layer in enumerate(self.layers):
            # Attention
            attn_params = sum(p.numel() for p in layer.attention.parameters())
            attn_norm_params = sum(p.numel() for p in layer.attention_norm.parameters())

            # MoE (Mixtral 방식: Router + Experts가 FFN 대체)
            moe = getattr(layer, 'moe', None)
            if moe is not None and hasattr(moe, 'router') and hasattr(moe, 'experts'):
                # Router
                router_params = sum(p.numel() for p in moe.router.parameters())
                router_total += router_params

                # Experts
                layer_expert_params = 0
                for expert in moe.experts:
                    layer_expert_params += sum(p.numel() for p in expert.parameters())
                expert_total += layer_expert_params

                ffn_norm_params = sum(p.numel() for p in layer.ffn_norm.parameters())

                layer_params = (attn_params + attn_norm_params +
                              router_params + ffn_norm_params + layer_expert_params)
            else:
                layer_params = attn_params + attn_norm_params

            layer_total += layer_params

            if layer_idx < 2 or layer_idx == len(self.layers) - 1:
                print(f"   🔹 Layer {layer_idx}:")
                print(f"      - Attention: {attn_params / 1e6:.2f}M")
                if moe is not None and hasattr(moe, 'router') and hasattr(moe, 'experts'):
                    print(f"      - Router: {router_params / 1e3:.1f}K")
                    print(f"      - Experts ({len(moe.experts)}개): {layer_expert_params / 1e6:.2f}M")

        if len(self.layers) > 3:
            print(f"   ... ({len(self.layers) - 3}개 레이어 생략)")

        total_params += layer_total
        print(f"   📊 레이어 총합 ({len(self.layers)}개): {layer_total / 1e6:.2f}M")
        print(f"      - Router 총합: {router_total / 1e6:.2f}M")
        print(f"      - Expert 총합: {expert_total / 1e6:.2f}M")

        # 출력 레이어
        norm_params = sum(p.numel() for p in self.norm.parameters())
        output_params = sum(p.numel() for p in self.output.parameters())
        total_params += norm_params + output_params
        print(f"   📤 출력 레이어: {output_params / 1e6:.2f}M")

        print(f"\n   ✅ 총 파라미터: {total_params / 1e6:.2f}M")
        return total_params

# ==========================================
# 2. Master 모델을 MoE로 변환
# ==========================================
def convert_master_to_moe(
    master_model: MasterModel,
    num_experts: int = 4,
    expert_init_method: str = "copy",
    new_expert_noise: float = 0.01
):
    """
    Master 모델을 MoE 모델로 변환
    
    Args:
        master_model: 학습된 Master 모델
        num_experts: Expert 수
        expert_init_method: Expert 초기화 방법 ("copy", "random", "copy_with_noise")
        new_expert_noise: 새 Expert에 추가할 노이즈 (copy_with_noise 사용 시)
    """
    import copy as _copy
    config = _copy.copy(master_model.config)
    config.num_experts = num_experts
    config.num_experts_per_tok = min(2, num_experts)  # 최대 2개 Expert 활성화
    
    # MoE 모델 생성
    moe_model = MoEModel(config, use_flash_attention=False)
    
    # Master 모델의 가중치 복사
    print(">>> Master 모델 가중치 복사 중...")
    
    # 임베딩 복사
    moe_model.token_embedding.load_state_dict(master_model.token_embedding.state_dict())
    
    # 레이어별 복사
    for layer_idx, (master_layer, moe_layer) in enumerate(zip(master_model.layers, moe_model.layers)):
        # Attention 복사
        moe_layer.attention.load_state_dict(master_layer.attention.state_dict())
        moe_layer.attention_norm.load_state_dict(master_layer.attention_norm.state_dict())

        # FFN Norm 복사
        master_ffn = master_layer.ffn
        moe_layer.ffn_norm.load_state_dict(master_layer.ffn_norm.state_dict())

        # Expert 접근: layer.moe.experts
        experts = moe_layer.moe.experts

        # Master FFN을 Expert들로 복제 (Sparse Upcycling)
        if expert_init_method == "copy":
            # 첫 번째 Expert: Master FFN 복사
            experts[0].load_state_dict(master_ffn.state_dict())

            # 나머지 Expert들: 첫 번째 Expert 복사
            for expert_idx in range(1, num_experts):
                experts[expert_idx].load_state_dict(master_ffn.state_dict())
                print(f"   -> Layer {layer_idx}, Expert {expert_idx}: Master FFN 복사")

        elif expert_init_method == "copy_with_noise":
            # 모든 Expert를 Master FFN에서 초기화 (Upcycling with Noise Injection)
            for expert_idx in range(num_experts):
                # Master FFN 가중치 복사 (Clone)
                experts[expert_idx].load_state_dict(master_ffn.state_dict())

                # Symmetry Breaking을 위한 노이즈 주입
                master_params = dict(master_ffn.named_parameters())
                with torch.no_grad():
                    for param_name, param in experts[expert_idx].named_parameters():
                        master_param = master_params.get(param_name)
                        if master_param is not None:
                            noise_scale = master_param.data.std().item() * new_expert_noise
                            param.data += torch.randn_like(param.data) * noise_scale

                print(f"   -> Layer {layer_idx}, Expert {expert_idx}: Master FFN 복사 + 노이즈 (std의 {new_expert_noise*100:.1f}%)")

        elif expert_init_method == "random":
            # 첫 번째 Expert만 Master FFN 복사
            experts[0].load_state_dict(master_ffn.state_dict())
            # 나머지는 랜덤 초기화 (이미 초기화됨)
            print(f"   -> Layer {layer_idx}, Expert 0: Master FFN 복사, 나머지: 랜덤")
    
    # 출력 레이어 복사
    moe_model.norm.load_state_dict(master_model.norm.state_dict())
    moe_model.output.load_state_dict(master_model.output.state_dict())
    
    print(f"   -> ✅ Master 모델을 {num_experts}개 Expert MoE 모델로 변환 완료")
    
    return moe_model

# ==========================================
# 3. MoE 모델 학습 함수
# ==========================================
def train_moe_model(
    model, 
    dataloader, 
    optimizer, 
    criterion,
    device,
    num_epochs: int = 3,
    save_dir: str = "moe_model_checkpoints",
    use_mixed_precision: bool = True,
    save_every_n_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    train_router_only: bool = False,
    train_experts_only: bool = False,
):
    """MoE 모델 학습 실행"""
    print(f"\n>>> MoE 모델 학습 시작!")
    print(f"   -> Epochs: {num_epochs}")
    print(f"   -> Router만 학습: {train_router_only}")
    print(f"   -> Expert만 학습: {train_experts_only}")
    
    model.train()
    scaler = torch.amp.GradScaler('cuda') if use_mixed_precision and device.type == "cuda" else None
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    os.makedirs(save_dir, exist_ok=True)
    all_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"MoE Model, Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (input_ids, labels) in enumerate(progress_bar):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            if scaler is not None:
                with torch.amp.autocast('cuda', dtype=dtype):
                    logits, aux_loss = model(input_ids)
                    task_loss = criterion(logits.view(-1, model.config.vocab_size), labels.view(-1))
                    total_loss = task_loss + (0.01 * aux_loss)
                    loss = total_loss / gradient_accumulation_steps
            else:
                logits, aux_loss = model(input_ids)
                task_loss = criterion(logits.view(-1, model.config.vocab_size), labels.view(-1))
                total_loss = task_loss + (0.01 * aux_loss)
                loss = total_loss / gradient_accumulation_steps
            
            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
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
            
            # 메모리 정리는 OOM 방지용으로 드물게 실행
            if batch_idx % 1000 == 0 and batch_idx > 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # 진행 상황 업데이트
            if batch_idx % 20 == 0:
                postfix_dict = {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{epoch_loss/(batch_idx+1):.4f}",
                    "mem": f"{get_gpu_memory()['allocated']:.1f}GB" if get_gpu_memory() else "N/A"
                }
                # aux_loss 정보 추가 (있는 경우)
                if 'aux_loss' in locals() and aux_loss is not None:
                    postfix_dict["aux_loss"] = f"{aux_loss.item():.6f}"
                progress_bar.set_postfix(postfix_dict)
        
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
            checkpoint_path = os.path.join(save_dir, f"moe_model_epoch_{epoch+1}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': model.config,
                'epoch': epoch + 1,
                'loss': avg_epoch_loss,
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"   -> 체크포인트 저장: {checkpoint_path}")
    
    return all_losses

def train_moe_stage(
    stage_name: str,
    moe_model: MoEModel,
    dataloader: DataLoader,
    device: torch.device,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    train_router_only: bool = False,
    train_experts_only: bool = False,
    save_dir: str = "moe_model_checkpoints",
    gradient_accumulation_steps: int = 4,
    freeze_attention: bool = True,
):
    """단계별 MoE 학습"""
    print(f"\n{'='*60}")
    print(f"Stage: {stage_name}")
    print(f"{'='*60}")

    # 학습 모드 설정
    if freeze_attention:
        moe_model.freeze_attention()
    else:
        moe_model.unfreeze_attention()

    if train_router_only:
        moe_model.freeze_experts()
        moe_model.unfreeze_routers()
        print(f">>> 학습 모드: Router만 학습 (Attention + Expert 동결)")
    elif train_experts_only:
        moe_model.freeze_routers()
        moe_model.unfreeze_experts()
        print(f">>> 학습 모드: Expert만 학습 (Attention + Router 동결)")
    else:
        moe_model.unfreeze_experts()
        moe_model.unfreeze_routers()
        print(f">>> 학습 모드: 전체 MoE 학습 (Router + Expert, Attention 동결)")
    
    # 학습 가능한 파라미터 확인
    trainable_params = sum(p.numel() for p in moe_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in moe_model.parameters())
    print(f"   -> 학습 가능한 파라미터: {trainable_params / 1e6:.2f}M / {total_params / 1e6:.2f}M ({100 * trainable_params / total_params:.2f}%)")
    
    # Loss 함수
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # 옵티마이저
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            moe_model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        )
        print(f"   -> ✅ 8-bit AdamW 사용 (lr={learning_rate})")
    except ImportError:
        optimizer = torch.optim.AdamW(
            moe_model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        )
        print(f"   -> ✅ AdamW 사용 (lr={learning_rate})")
    
    # 학습
    losses = train_moe_model(
        model=moe_model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=num_epochs,
        save_dir=save_dir,
        save_every_n_epochs=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        train_router_only=train_router_only,
        train_experts_only=train_experts_only,
    )
    
    # 체크포인트 저장
    checkpoint_path = os.path.join(save_dir, f"moe_{stage_name.lower().replace(' ', '_')}.pt")
    torch.save({
        'model_state_dict': moe_model.state_dict(),
        'config': moe_model.config,
        'losses': losses,
        'final_loss': losses[-1] if losses else None,
        'stage': stage_name,
        'train_router_only': train_router_only,
        'train_experts_only': train_experts_only,
    }, checkpoint_path)
    print(f"\n>>> Stage 체크포인트 저장: {checkpoint_path}")
    
    return losses, checkpoint_path

def main():
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 디바이스: {device}")
    if torch.cuda.is_available():
        print(f"   -> GPU: {torch.cuda.get_device_name(0)}")
        print(f"   -> VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # ==========================================
    # Master 모델 로드
    # ==========================================
    moe_path = "master_model_checkpoints/master_model_for_moe.pt"
    epoch_path = "master_model_checkpoints/master_model_epoch_1.pt"
    if os.path.exists(moe_path):
        print(f"\n>>> Master 모델 로드 중: {moe_path} (MoE용 full-precision)")
        import warnings
        warnings.warn(
            f"weights_only=False로 {moe_path}를 로드합니다. "
            "신뢰할 수 없는 파일이라면 보안 위험이 있습니다.",
            UserWarning,
            stacklevel=2,
        )
        ckpt = torch.load(moe_path, map_location='cpu', weights_only=False)
        master_model = MasterModel(ckpt['config'], use_flash_attention=False)
        # QLoRA 저장 시 LoRA/base_layer 키가 포함될 수 있으므로 필터링
        state = {k: v for k, v in ckpt['model_state_dict'].items()
                 if 'lora_' not in k and 'base_layer' not in k}
        state = convert_legacy_state_dict(state)
        master_model.load_state_dict(state, strict=True)
        del ckpt, state; gc.collect()
    elif os.path.exists(epoch_path):
        print(f"\n>>> Master 모델 로드 중: {epoch_path}")
        master_model = load_master_for_moe(epoch_path, device='cpu')
    else:
        print(f"   -> ❌ Master 체크포인트를 찾을 수 없습니다.")
        print(f"   -> master_model_for_moe.pt 또는 master_model_epoch_1.pt 필요")
        print(f"   -> train_master.py를 실행하여 Master 모델을 학습하세요.")
        return
    print(f"   -> ✅ Master 모델 로드 완료")
    
    # ==========================================
    # MoE 설정 (24GB GPU: num_experts 2 권장, 여유 있으면 4)
    # ==========================================
    num_experts = 2  # Expert 수 (24GB: 2, 32GB+: 4)
    expert_init_method = "copy_with_noise"  # "copy", "copy_with_noise", "random"
    
    print(f"\n>>> MoE 설정:")
    print(f"   -> Expert 수: {num_experts}")
    print(f"   -> Expert 초기화: {expert_init_method}")
    
    # ==========================================
    # Master 모델을 MoE로 변환
    # ==========================================
    print(f"\n>>> Master 모델을 MoE로 변환 중...")
    moe_model = convert_master_to_moe(
        master_model,
        num_experts=num_experts,
        expert_init_method=expert_init_method,
        new_expert_noise=0.01
    )
    # Master 모델 메모리 해제 (24GB GPU OOM 방지)
    del master_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # bf16으로 변환하여 모델 메모리 절반으로 절약 (FP32: ~17GB → bf16: ~8.5GB)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    moe_model = moe_model.to(device=device, dtype=compute_dtype)
    print(f"   -> 모델 dtype: {compute_dtype} (메모리 절약)")
    # Gradient checkpointing으로 VRAM 절약
    for layer in moe_model.layers:
        if hasattr(layer, "use_gradient_checkpointing"):
            layer.use_gradient_checkpointing = True
    moe_model.print_param_stats()
    
    # ==========================================
    # 데이터셋 준비 (Data Mixing: 70% 나무위키 + 30% 코드)
    # ==========================================
    print("\n>>> 데이터셋 로딩 중...")
    
    # 메인 데이터셋: 나무위키 (70%)
    print("   -> 나무위키 데이터셋 로딩 중...")
    try:
        ds_namu = load_dataset("heegyu/namuwiki-extracted", split="train[:10000]")
        namu_texts = [item["text"].strip() for item in ds_namu if item.get("text", "").strip()]
        del ds_namu; gc.collect()
        print(f"   -> 나무위키 텍스트 수: {len(namu_texts)}")
    except Exception as e:
        print(f"   -> ⚠️ 나무위키 데이터셋 로딩 실패: {e}")
        namu_texts = ["안녕하세요. 이것은 나무위키 테스트 텍스트입니다."] * 1000
    
    # 서브 데이터셋: Python 코드 (30%)
    print("   -> Python 코드 데이터셋 로딩 중...")
    try:
        ds_code = load_dataset("flytech/python-codes-25k", split="train[:5000]")
        code_texts = [(item.get("text","") or item.get("content","") or item.get("code","")).strip()
                       for item in ds_code]
        code_texts = [t for t in code_texts if t]
        del ds_code; gc.collect()
        print(f"   -> Python 코드 텍스트 수: {len(code_texts)}")
    except Exception as e:
        print(f"   -> ⚠️ Python 코드 데이터셋 로딩 실패: {e}")
        code_texts = ["def hello(): print('Hello, World!')"] * 1000
    
    # 데이터셋 Mixing: 7:3 비율로 섞기
    print("\n>>> 데이터셋 Mixing (70% 나무위키 + 30% 코드)...")

    namu_ratio, code_ratio = 0.7, 0.3

    # 비율에 맞게 샘플 수 계산
    target_total = min(
        int(len(namu_texts) / namu_ratio),
        int(len(code_texts) / code_ratio),
        50000,
    )
    namu_samples = min(int(target_total * namu_ratio), len(namu_texts))
    code_samples = min(int(target_total * code_ratio), len(code_texts))

    # 랜덤 샘플링 + 셔플
    random.shuffle(namu_texts)
    random.shuffle(code_texts)
    mixed_texts = namu_texts[:namu_samples] + code_texts[:code_samples]
    random.shuffle(mixed_texts)
    
    print(f"   -> 나무위키 샘플: {namu_samples}개 ({namu_ratio*100:.0f}%)")
    print(f"   -> 코드 샘플: {code_samples}개 ({code_ratio*100:.0f}%)")
    print(f"   -> 총 혼합 텍스트 수: {len(mixed_texts)}")
    
    # 토크나이저 — Master와 동일한 어휘 사전 사용 (임베딩 ID 일치 필수)
    print("\n>>> 토크나이저 로드 중...")
    tokenizer_path = "master_model_checkpoints/tokenizer.json"
    if os.path.exists(tokenizer_path):
        tokenizer = KoreanBPETokenizer.load(tokenizer_path)
        print(f"   -> Master 토크나이저 로드: {tokenizer_path}")
    else:
        print(f"   -> ⚠️ 경고: Master 토크나이저({tokenizer_path}) 없음 — 임베딩 불일치 가능!")
        tokenizer = KoreanBPETokenizer(vocab_size=moe_model.config.vocab_size)
        vocab_texts = mixed_texts[:min(10000, len(mixed_texts))]
        tokenizer.build_vocab(vocab_texts)
        print(f"   -> 새 어휘 사전 구축 (샘플 수: {len(vocab_texts)})")
    
    # 데이터로더
    dataset = TextDataset(mixed_texts, tokenizer, max_length=moe_model.config.max_seq_len)
    batch_size = 1
    gradient_accumulation_steps = 4
    # 사전 토큰화 완료 → num_workers=0으로 충분 (fork 경고 방지)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"   -> 배치 크기: {batch_size}")
    print(f"   -> Gradient Accumulation Steps: {gradient_accumulation_steps}")
    print(f"   -> 배치 수: {len(dataloader)}")
    
    # ==========================================
    # 단계별 학습
    # ==========================================
    save_dir = "moe_model_checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    # Stage 1: Router만 학습 — 체크포인트가 있으면 건너뜀
    stage1_ckpt_path = os.path.join(save_dir, "moe_stage1_router.pt")
    stage1_losses = None
    if os.path.exists(stage1_ckpt_path):
        print("\n" + "="*60)
        print("📌 Stage 1: Router 학습 — 체크포인트 발견, 건너뜀")
        print("="*60)
        print(f"   -> 로드: {stage1_ckpt_path}")
        stage1_ckpt = torch.load(stage1_ckpt_path, map_location='cpu', weights_only=False)
        moe_model.load_state_dict(stage1_ckpt['model_state_dict'], strict=False)
        moe_model = moe_model.to(device=device, dtype=compute_dtype)
        stage1_losses = stage1_ckpt.get('losses', [])
        del stage1_ckpt; gc.collect(); torch.cuda.empty_cache()
        print("   -> ✅ Stage 1 체크포인트 로드 완료")
    else:
        print("\n" + "="*60)
        print("📌 Stage 1: Router 학습")
        print("="*60)
        print("목적: Router가 토큰을 적절한 Expert로 라우팅하도록 학습")
        print("설정: Attention + Expert 동결, Router만 학습")

        stage1_losses, stage1_checkpoint = train_moe_stage(
            stage_name="Stage1_Router",
            moe_model=moe_model,
            dataloader=dataloader,
            device=device,
            num_epochs=1,
            learning_rate=1e-4,
            train_router_only=True,
            save_dir=save_dir,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        print(f"\n✅ Stage 1 완료! 평균 Loss: {sum(stage1_losses) / len(stage1_losses):.4f}")

    # Stage 2: Expert 학습 (Router 고정)
    print("\n" + "="*60)
    print("📌 Stage 2: Expert 학습")
    print("="*60)
    print("목적: Expert들이 각자의 전문 분야로 분화하도록 학습")
    print("설정: Attention + Router 동결, Expert만 학습")

    stage2_losses, stage2_checkpoint = train_moe_stage(
        stage_name="Stage2_Expert",
        moe_model=moe_model,
        dataloader=dataloader,
        device=device,
        num_epochs=1,
        learning_rate=2e-4,  # Expert 학습은 중간 학습률
        train_experts_only=True,
        save_dir=save_dir,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    print(f"\n✅ Stage 2 완료! 평균 Loss: {sum(stage2_losses) / len(stage2_losses):.4f}")

    # Stage 3: 전체 미세조정 (Router + Expert 모두 학습)
    print("\n" + "="*60)
    print("📌 Stage 3: 전체 미세조정")
    print("="*60)
    print("목적: Router와 Expert를 함께 최적화 (Attention은 보존)")
    print("설정: Attention 동결, Router + Expert 학습")

    stage3_losses, stage3_checkpoint = train_moe_stage(
        stage_name="Stage3_Full",
        moe_model=moe_model,
        dataloader=dataloader,
        device=device,
        num_epochs=1,
        learning_rate=1e-4,  # 전체 학습은 작은 학습률로 안정적으로
        train_router_only=False,
        train_experts_only=False,
        save_dir=save_dir,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    
    print(f"\n✅ Stage 3 완료! 평균 Loss: {sum(stage3_losses) / len(stage3_losses):.4f}")
    
    # ==========================================
    # 최종 결과
    # ==========================================
    print("\n" + "="*60)
    print("🎉 MoE 단계별 학습 완료!")
    print("="*60)
    
    print(f"\n>>> 학습 결과 요약:")
    if stage1_losses:
        print(f"   Stage 1 (Router): {sum(stage1_losses) / len(stage1_losses):.4f}")
    else:
        print(f"   Stage 1 (Router): (체크포인트에서 로드)")
    print(f"   Stage 2 (Expert): {sum(stage2_losses) / len(stage2_losses):.4f}")
    print(f"   Stage 3 (Full):  {sum(stage3_losses) / len(stage3_losses):.4f}")
    
    # 최종 모델 저장
    final_path = os.path.join(save_dir, "moe_model_final.pt")
    torch.save({
        'model_state_dict': moe_model.state_dict(),
        'config': moe_model.config,
        'stage1_losses': stage1_losses or [],
        'stage2_losses': stage2_losses,
        'stage3_losses': stage3_losses,
        'final_loss': stage3_losses[-1] if stage3_losses else None,
        'num_experts': num_experts,
        'expert_init_method': expert_init_method,
    }, final_path)
    
    # 토크나이저 저장
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    
    print(f"\n>>> 최종 MoE 모델 저장: {final_path}")
    print(f"   -> tokenizer.json")
    print(f"\n💡 다음 단계: Progressive MoE로 Expert 수를 점진적으로 확장할 수 있습니다.")

if __name__ == "__main__":
    main()
