"""
점진적 MoE 학습을 위한 확장 가능한 MoE 레이어
- Main MoE 먼저 학습
- 나중에 추가 Expert들을 동적으로 추가
- Router 자동 확장
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import copy
import os

from model import ModelConfig, Expert, RMSNorm

# ==========================================
# 1. 확장 가능한 MoE 설정
# ==========================================
@dataclass
class ProgressiveMoEConfig:
    """점진적 MoE 설정"""
    # 기본 MoE 설정
    base_config: ModelConfig

    # 점진 학습 설정
    initial_num_experts: int = 2      # 초기 expert 수
    max_num_experts: int = 8          # 최대 expert 수
    experts_per_stage: int = 2        # 단계별 추가할 expert 수

    # 새 expert 초기화 방식
    new_expert_init: str = "random"   # "random", "copy_best", "interpolate"

    # Router 확장 방식
    router_expansion: str = "extend"  # "extend", "reinit"
    router_init_temp: float = 0.1     # 새 expert에 대한 router 초기 온도

# ==========================================
# 2. 확장 가능한 Router
# ==========================================
class ExpandableRouter(nn.Module):
    """동적으로 expert 수를 확장할 수 있는 라우터"""
    def __init__(self, config: ModelConfig, max_experts: int = None):
        super().__init__()
        self.dim = config.dim
        self.current_num_experts = config.num_experts
        self.max_experts = max_experts or config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # 최대 크기로 gate 초기화 (사용하지 않는 부분은 마스킹)
        self.gate = nn.Linear(config.dim, self.max_experts, bias=False)

        # 사용 중인 expert 마스크
        self.register_buffer('expert_mask', torch.ones(self.max_experts, dtype=torch.bool))
        self.expert_mask[config.num_experts:] = False

    def forward(self, x):
        # 전체 gate 출력
        logits = self.gate(x)  # (batch, seq_len, max_experts)

        # 현재 활성 expert만 사용
        if not self.expert_mask.all():
            # 비활성 expert는 -inf로 마스킹
            mask = self.expert_mask.unsqueeze(0).unsqueeze(0).expand_as(logits)
            logits = logits.masked_fill(~mask, float('-inf'))

        return logits

    def add_experts(self, num_new_experts: int, init_method: str = "random"):
        """새로운 expert들을 위한 router 가중치 추가"""
        if self.current_num_experts + num_new_experts > self.max_experts:
            raise ValueError(f"Cannot add {num_new_experts} experts. "
                           f"Current: {self.current_num_experts}, Max: {self.max_experts}")

        # expert 마스크 업데이트
        start_idx = self.current_num_experts
        end_idx = start_idx + num_new_experts
        self.expert_mask[start_idx:end_idx] = True

        # 새로운 gate 가중치 초기화
        with torch.no_grad():
            if init_method == "random":
                # 랜덤 초기화 (작은 값)
                nn.init.normal_(self.gate.weight[start_idx:end_idx], mean=0.0, std=0.02)
            elif init_method == "zero":
                # 0으로 초기화 (점진적 학습에 유리)
                self.gate.weight[start_idx:end_idx].zero_()
            elif init_method == "copy_existing":
                # 기존 expert의 평균으로 초기화
                existing_weights = self.gate.weight[:self.current_num_experts]
                avg_weight = existing_weights.mean(dim=0, keepdim=True)
                self.gate.weight[start_idx:end_idx] = avg_weight + torch.randn_like(
                    self.gate.weight[start_idx:end_idx]) * 0.01

        self.current_num_experts += num_new_experts
        print(f"   -> Router expanded: {self.current_num_experts} experts")

# ==========================================
# 3. 점진적 MoE 레이어
# ==========================================
class ProgressiveMoELayer(nn.Module):
    """점진적으로 expert를 추가할 수 있는 MoE 레이어"""
    def __init__(self, config: ModelConfig, progressive_config: ProgressiveMoEConfig):
        super().__init__()
        self.config = config
        self.progressive_config = progressive_config

        # [P-2] config.num_experts를 initial_num_experts로 덮어쓴 복사본으로 Router 초기화
        init_config = copy.copy(config)
        init_config.num_experts = progressive_config.initial_num_experts

        # 확장 가능한 라우터
        self.router = ExpandableRouter(
            init_config,
            max_experts=progressive_config.max_num_experts
        )

        # Expert들 (처음에는 initial_num_experts개만 생성)
        self.experts = nn.ModuleList()
        self.current_num_experts = 0

        # [P-3] 실제 라우팅 카운트 누적 버퍼
        self.register_buffer(
            'routing_counts',
            torch.zeros(progressive_config.max_num_experts)
        )

        # 초기 expert들 추가 (Router는 이미 initial_num_experts로 초기화됨, 확장 불필요)
        for i in range(progressive_config.initial_num_experts):
            self.experts.append(Expert(config))
            self.current_num_experts += 1

        self.num_experts_per_tok = config.num_experts_per_tok

    def _create_expert(self, init_method: str) -> Expert:
        """초기화 방식에 따라 단일 Expert 생성"""
        if init_method == "copy_best":
            if len(self.experts) > 0:
                counts = self.routing_counts[:self.current_num_experts]
                best_idx = counts.argmax().item() if counts.sum() > 0 else 0
                new_expert = copy.deepcopy(self.experts[best_idx])
                for param in new_expert.parameters():
                    param.data += torch.randn_like(param.data) * 0.01
                return new_expert
        elif init_method == "interpolate":
            if len(self.experts) >= 2:
                new_expert = Expert(self.config)
                with torch.no_grad():
                    for (name1, param1), (name2, param2), (name_new, param_new) in zip(
                        self.experts[0].named_parameters(),
                        self.experts[1].named_parameters(),
                        new_expert.named_parameters()
                    ):
                        param_new.data = (param1.data + param2.data) / 2
                        param_new.data += torch.randn_like(param_new.data) * 0.01
                return new_expert
        return Expert(self.config)

    def add_experts(self, num_new_experts: int, init_method: str = None):
        """새로운 expert들 추가 (원자적: 전체 생성 후 한꺼번에 추가)"""
        if init_method is None:
            init_method = self.progressive_config.new_expert_init

        print(f">>> Adding {num_new_experts} new experts...")

        # 전체 experts 생성 (실패 시 기존 상태 보존)
        new_experts = [self._create_expert(init_method) for _ in range(num_new_experts)]

        # Router 먼저 확장 (ValueError 발생 시 experts 미추가)
        router_init = "zero" if init_method == "random" else "copy_existing"
        self.router.add_experts(num_new_experts, init_method=router_init)

        # 성공 시 experts 추가
        for e in new_experts:
            self.experts.append(e)
        self.current_num_experts += num_new_experts

        print(f"   -> Total experts: {self.current_num_experts}")

    def freeze_experts(self, expert_indices: List[int]):
        """특정 expert들을 동결 (기존 expert 보존)"""
        for idx in expert_indices:
            if idx < len(self.experts):
                for param in self.experts[idx].parameters():
                    param.requires_grad = False
        print(f"   -> Frozen experts: {expert_indices}")

    def unfreeze_experts(self, expert_indices: List[int]):
        """특정 expert들 해제"""
        for idx in expert_indices:
            if idx < len(self.experts):
                for param in self.experts[idx].parameters():
                    param.requires_grad = True
        print(f"   -> Unfrozen experts: {expert_indices}")

    def forward(self, x):
        bsz, seq_len, dim = x.shape

        # 라우터가 각 토큰에 대한 전문가 점수 계산
        router_logits = self.router(x)  # (bsz, seq_len, max_experts)

        # Top-k 전문가 선택 (현재 활성 expert 수 고려)
        k = min(self.num_experts_per_tok, self.current_num_experts)
        top_k_logits, top_k_indices = torch.topk(router_logits, k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)  # (bsz, seq_len, k)

        # [P-3] 실제 라우팅 카운트 누적
        with torch.no_grad():
            flat = top_k_indices.view(-1)
            self.routing_counts.scatter_add_(0, flat, flat.new_ones(flat.numel(), dtype=torch.float))

        # 평탄화
        flat_x = x.view(-1, dim)
        flat_top_k_indices = top_k_indices.view(-1, k)
        flat_top_k_gates = top_k_gates.view(-1, k)

        # 출력 초기화
        flat_output = torch.zeros_like(flat_x)

        # 벡터화 dispatch: O(E) 단일 루프
        T = flat_x.shape[0]
        tokens_for_dispatch = flat_x.unsqueeze(1).expand(-1, k, -1).reshape(T * k, dim)
        expert_ids = flat_top_k_indices.reshape(T * k)
        weights = flat_top_k_gates.reshape(T * k, 1)

        for expert_idx in range(self.current_num_experts):
            mask = (expert_ids == expert_idx)
            if not mask.any():
                continue
            out = self.experts[expert_idx](tokens_for_dispatch[mask])
            token_indices = torch.where(mask)[0] % T
            flat_output.scatter_add_(
                0,
                token_indices.unsqueeze(1).expand(-1, dim),
                out * weights[mask],
            )

        output = flat_output.view(bsz, seq_len, dim)

        # [P-1] aux_loss 계산 — 활성 expert 범위의 logits만 사용
        active_logits = router_logits[:, :, :self.current_num_experts]
        aux_loss = self._load_balancing_loss(active_logits, top_k_indices)

        return output, aux_loss

    def _load_balancing_loss(self, router_logits, top_k_indices):
        """Switch Transformer style load balancing loss"""
        E = self.current_num_experts
        routing_probs = F.softmax(router_logits, dim=-1)
        avg_prob = routing_probs.mean(dim=[0, 1])          # (E,)
        # clamp로 인덱스 범위 초과 방지 (ExpandableRouter 마스킹 이후에도 안전)
        flat_indices = top_k_indices.view(-1).clamp(max=E - 1)
        expert_counts = torch.zeros(E, device=router_logits.device)
        expert_counts.scatter_add_(
            0, flat_indices,
            flat_indices.new_ones(flat_indices.numel(), dtype=torch.float)
        )
        fraction = expert_counts / flat_indices.numel()
        return E * (fraction * avg_prob).sum()

    def get_expert_usage_stats(self):
        """각 expert의 실제 라우팅 비율 반환"""
        counts = self.routing_counts[:self.current_num_experts].clone()
        total = counts.sum().clamp(min=1.0)
        return (counts / total).cpu().numpy()

# ==========================================
# 4. 점진적 학습을 위한 유틸리티
# ==========================================
class ProgressiveMoETrainer:
    """점진적 MoE 학습 관리자"""
    def __init__(self, model, progressive_config: ProgressiveMoEConfig):
        self.model = model
        self.config = progressive_config
        self.current_stage = 0
        self.training_history = []

    def expand_model(self, stage: int = None):
        """모델에 새로운 expert들 추가"""
        if stage is None:
            stage = self.current_stage + 1

        print(f"\n>>> Stage {stage}: Expanding MoE...")

        # 각 레이어의 MoE 확장
        for layer_idx, layer in enumerate(self.model.layers):
            if hasattr(layer, 'moe') and isinstance(layer.moe, ProgressiveMoELayer):
                # 기존 expert들을 동결 (선택적)
                existing_experts = list(range(layer.moe.current_num_experts))
                if stage > 1:  # 첫 확장이 아닌 경우
                    layer.moe.freeze_experts(existing_experts[:-self.config.experts_per_stage])

                # 새 expert들 추가
                layer.moe.add_experts(self.config.experts_per_stage)

                print(f"   -> Layer {layer_idx}: {layer.moe.current_num_experts} experts")

        self.current_stage = stage

    def save_stage_checkpoint(self, save_dir: str, stage: int, additional_info: dict = None):
        """단계별 체크포인트 저장"""
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'stage': stage,
            'training_history': self.training_history,
        }
        if additional_info:
            checkpoint.update(additional_info)

        checkpoint_path = os.path.join(save_dir, f"progressive_stage_{stage}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"   -> Stage {stage} checkpoint saved: {checkpoint_path}")

    def load_stage_checkpoint(self, checkpoint_path: str):
        """단계별 체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_stage = checkpoint.get('stage', 0)
        self.training_history = checkpoint.get('training_history', [])
        print(f"   -> Loaded stage {self.current_stage} checkpoint")
        return checkpoint

# ==========================================
# 5. 테스트 코드
# ==========================================
if __name__ == "__main__":
    # 설정
    base_config = ModelConfig(
        vocab_size=32000,
        dim=1024,
        n_layers=4,
        n_heads=8,
        head_dim=128,
        num_experts=2,  # 초기값 (실제로는 progressive_config가 제어)
        num_experts_per_tok=2,
    )

    progressive_config = ProgressiveMoEConfig(
        base_config=base_config,
        initial_num_experts=2,
        max_num_experts=8,
        experts_per_stage=2,
        new_expert_init="copy_best",
    )

    # 테스트
    print("=== Progressive MoE Layer 테스트 ===")
    moe_layer = ProgressiveMoELayer(base_config, progressive_config)

    # 초기 상태
    print(f"초기 experts: {moe_layer.current_num_experts}")

    # 테스트 입력
    x = torch.randn(2, 128, base_config.dim)
    output1, aux_loss1 = moe_layer(x)
    print(f"초기 출력 shape: {output1.shape}, aux_loss: {aux_loss1.item():.6f}")

    # Expert 추가
    moe_layer.add_experts(2)
    output2, aux_loss2 = moe_layer(x)
    print(f"확장 후 출력 shape: {output2.shape}, aux_loss: {aux_loss2.item():.6f}")
    print(f"최종 experts: {moe_layer.current_num_experts}")

    # 사용량 통계
    usage_stats = moe_layer.get_expert_usage_stats()
    print(f"Expert 사용량: {usage_stats}")

    print("✅ Progressive MoE 테스트 완료!")
