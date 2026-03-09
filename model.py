import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

# ==========================================
# 1. 모델 설정
# ==========================================
@dataclass
class ModelConfig:
    # 기본 모델 설정
    vocab_size: int = 50000
    dim: int = 4096              # 모델 차원
    n_layers: int = 32           # 전체 레이어 수
    n_heads: int = 32            # 어텐션 헤드 수
    head_dim: int = 128          # 헤드 차원
    max_seq_len: int = 2048      # 최대 시퀀스 길이
    multiple_of: int = 256       # FFN 히든 차원 배수
    dropout: float = 0.1

    # MoE 설정
    num_experts: int = 8          # 전문가(Expert) 개수
    num_experts_per_tok: int = 2 # 토큰당 활성화할 전문가 수
    # TODO: expert_capacity_factor 미사용 — 토큰 드롭 미구현
    expert_capacity_factor: float = 1.25

# ==========================================
# 2. 기본 컴포넌트 (RMSNorm, RoPE)
# ==========================================
class RMSNorm(nn.Module):
    """RMS 정규화"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # float32로 캐스팅하여 bf16/fp16 오버플로 방지 (LLaMA 방식)
        input_dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).to(input_dtype)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """RoPE 주파수 사전 계산"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """RoPE 적용"""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:xq.shape[1]].unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# ==========================================
# 3. Attention 구현
# ==========================================
class Attention(nn.Module):
    def __init__(self, config: ModelConfig, use_flash_attention=False):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.dim = config.dim
        self.use_flash_attention = use_flash_attention

        self.wq = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)
        self.dropout = config.dropout

        # FlashAttention 사용 가능 여부 확인
        if use_flash_attention:
            try:
                from flash_attn import flash_attn_func
                self.flash_attn_func = flash_attn_func
                self.use_flash_attention = True
            except ImportError:
                self.use_flash_attention = False
                print("   -> ⚠️ FlashAttention 미설치, scaled_dot_product_attention 사용")

    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        # FlashAttention 사용 (학습 + 추론 모두)
        if self.use_flash_attention:
            # FlashAttention은 (batch, seq, heads, head_dim) 형태 필요
            output = self.flash_attn_func(
                xq, xk, xv,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True
            )
            output = output.contiguous().view(bsz, seqlen, -1)
        else:
            # PyTorch의 scaled_dot_product_attention 사용
            xq = xq.transpose(1, 2)
            xk = xk.transpose(1, 2)
            xv = xv.transpose(1, 2)
            
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
            
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        return self.wo(output)

# ==========================================
# 4. MoE (Mixture of Experts) 구현
# ==========================================
class Expert(nn.Module):
    """단일 전문가 네트워크 (SwiGLU FFN)"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        
        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)  # Gate
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)  # Down
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)  # Up

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Router(nn.Module):
    """라우터: 토큰을 적절한 전문가로 라우팅"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(config.dim, config.num_experts, bias=False)
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

    def forward(self, x):
        # (batch, seq_len, num_experts)
        logits = self.gate(x)
        return logits

class MoELayer(nn.Module):
    """MoE 레이어: 여러 전문가와 라우터 (배치 처리 최적화)"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.router = Router(config)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])
        self.num_experts_per_tok = config.num_experts_per_tok

    def forward(self, x):
        bsz, seq_len, dim = x.shape
        
        # 라우터가 각 토큰에 대한 전문가 점수 계산
        router_logits = self.router(x)  # (bsz, seq_len, num_experts)

        # 전체 softmax 한 번만 계산 (gating + aux_loss 공유)
        routing_probs = F.softmax(router_logits, dim=-1)

        # Top-k 전문가 선택
        top_k_gates, top_k_indices = torch.topk(routing_probs, self.num_experts_per_tok, dim=-1)
        # top-k 후 재정규화
        top_k_gates = top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)
        
        # 평탄화: (bsz * seq_len, dim)
        flat_x = x.view(-1, dim)  # (bsz * seq_len, dim)
        flat_top_k_indices = top_k_indices.view(-1, self.num_experts_per_tok)  # (bsz * seq_len, k)
        flat_top_k_gates = top_k_gates.view(-1, self.num_experts_per_tok)  # (bsz * seq_len, k)
        
        # 출력 초기화
        flat_output = torch.zeros_like(flat_x)  # (bsz * seq_len, dim)

        # 벡터화 dispatch: O(E) 단일 루프 (기존 O(k×E) 중첩 루프 제거)
        T = flat_x.shape[0]
        k = self.num_experts_per_tok
        tokens_for_dispatch = flat_x.unsqueeze(1).expand(-1, k, -1).reshape(T * k, dim)
        expert_ids = flat_top_k_indices.reshape(T * k)
        weights = flat_top_k_gates.reshape(T * k, 1)

        for expert_idx in range(self.config.num_experts):
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
        
        # 원래 형태로 복원
        output = flat_output.view(bsz, seq_len, dim)

        # Load balancing auxiliary loss
        aux_loss = self._load_balancing_loss(routing_probs, top_k_indices)

        return output, aux_loss

    def _load_balancing_loss(self, routing_probs, top_k_indices):
        """Switch Transformer style load balancing loss.

        aux_loss = N * sum_i(f_i * P_i)
        - f_i: fraction of tokens dispatched to expert i
        - P_i: average routing probability for expert i
        """
        num_experts = self.config.num_experts
        avg_prob = routing_probs.mean(dim=[0, 1])          # (E,)

        # 벡터화: view(-1) 한 번으로 처리 (루프 제거)
        flat_indices = top_k_indices.view(-1)               # (bsz*seq*k,)
        expert_counts = torch.zeros(num_experts, device=routing_probs.device)
        expert_counts.scatter_add_(
            0, flat_indices,
            flat_indices.new_ones(flat_indices.numel(), dtype=torch.float)
        )
        fraction = expert_counts / flat_indices.numel()

        return num_experts * (fraction * avg_prob).sum()

# ==========================================
# 5. Transformer 블록
# ==========================================
class MasterBlock(nn.Module):
    """Master 블록: Attention + 단일 FFN (MoE 없음)"""
    def __init__(self, config: ModelConfig, use_flash_attention=False):
        super().__init__()
        self.attention = Attention(config, use_flash_attention=use_flash_attention)
        self.ffn = Expert(config)  # 단일 FFN 사용
        self.attention_norm = RMSNorm(config.dim)
        self.ffn_norm = RMSNorm(config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.use_gradient_checkpointing = False  # 기본값
        # aux_loss=0 스칼라 캐시 (매 forward마다 생성하지 않음)
        self.register_buffer('_zero_aux_loss', torch.tensor(0.0), persistent=False)

    def forward(self, x, freqs_cis):
        # Gradient Checkpointing 지원
        if self.use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, freqs_cis, use_reentrant=False
            )
        else:
            return self._forward_impl(x, freqs_cis)
    
    def _forward_impl(self, x, freqs_cis):
        # Pre-norm 구조
        # Attention
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        h = self.dropout(h)

        # FFN (MoE 없이 단일 Expert 사용)
        out = h + self.ffn(self.ffn_norm(h))
        out = self.dropout(out)

        return out, self._zero_aux_loss

class TransformerBlock(nn.Module):
    """Transformer 블록: Attention + MoE"""
    def __init__(self, config: ModelConfig, use_flash_attention=False):
        super().__init__()
        self.config = config
        self.attention = Attention(config, use_flash_attention=use_flash_attention)
        self.moe = MoELayer(config)
        self.attention_norm = RMSNorm(config.dim)
        self.ffn_norm = RMSNorm(config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.use_gradient_checkpointing = False  # 기본값

    def forward(self, x, freqs_cis):
        # Gradient Checkpointing 지원
        if self.use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, freqs_cis, use_reentrant=False
            )
        else:
            return self._forward_impl(x, freqs_cis)

    def _forward_impl(self, x, freqs_cis):
        # Pre-norm 구조
        # Attention
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        h = self.dropout(h)

        # MoE
        moe_output, aux_loss = self.moe(self.ffn_norm(h))
        out = h + moe_output
        out = self.dropout(out)

        return out, aux_loss

# MoEBlock은 TransformerBlock으로 통합됨 (하위 호환용 별칭)
MoEBlock = TransformerBlock

# ==========================================
# 6. 전체 Transformer MoE 모델
# ==========================================
class TransformerMoE(nn.Module):
    """Transformer + MoE 전체 모델"""
    def __init__(self, config: ModelConfig, use_flash_attention=False):
        super().__init__()
        self.config = config
        
        # 임베딩
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        
        # 레이어
        self.layers = nn.ModuleList([
            TransformerBlock(config, use_flash_attention=use_flash_attention) for _ in range(config.n_layers)
        ])
        
        # 출력
        self.norm = RMSNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # RoPE 주파수 사전 계산 (register_buffer로 자동 device 연동)
        self.register_buffer(
            'freqs_cis',
            precompute_freqs_cis(config.head_dim, config.max_seq_len * 2),
            persistent=False,
        )

        # Residual projection 마킹 후 가중치 초기화
        self._mark_residual_projections()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # Residual projection (wo, w2)에는 depth-scaled init (GPT-2 방식)
            if hasattr(module, '_is_residual'):
                std = 0.02 / math.sqrt(2 * self.config.n_layers)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _mark_residual_projections(self):
        """Residual projection 레이어에 _is_residual 마커 설정"""
        for layer in self.layers:
            layer.attention.wo._is_residual = True
            if hasattr(layer, 'moe'):
                for expert in layer.moe.experts:
                    expert.w2._is_residual = True

    def forward(self, tokens):
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

    def get_num_params(self):
        """모델 파라미터 수 계산"""
        return sum(p.numel() for p in self.parameters())

# 하위 호환용 별칭
MasterBlockWithRouterMoE = TransformerBlock
JambaConfig = ModelConfig
JambaBlock = TransformerBlock
Jamba2 = TransformerMoE

# ==========================================
# 7. 테스트 코드
# ==========================================
if __name__ == "__main__":
    config = ModelConfig(
        vocab_size=50000,
        dim=2048,  # 테스트용 작은 크기
        n_layers=4,
        n_heads=16,
        head_dim=128,
        max_seq_len=512,
        num_experts=4,
        num_experts_per_tok=2,
    )

    model = TransformerMoE(config)
    print(f"Transformer MoE 모델 생성 완료!")
    print(f"파라미터 수: {model.get_num_params() / 1e6:.1f}M")
    
    # 테스트
    dummy_input = torch.randint(0, 50000, (2, 128))
    try:
        result = model(dummy_input)
        if isinstance(result, tuple):
            output, aux_loss = result
            print(f"🧪 입력: {dummy_input.shape}")
            print(f"🧪 출력: {output.shape}, aux_loss: {aux_loss.item():.6f}")
        else:
            output = result
            print(f"🧪 입력: {dummy_input.shape}")
            print(f"🧪 출력: {output.shape}")
        print("✅ Forward Pass 성공!")
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()
