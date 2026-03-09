# MoE 프로젝트 코드 리뷰 가이드라인

> **검토 파일**: `model.py` · `train_master.py` · `train_moe.py` · `utils.py` · `progressive_moe_layer.py` · `test_progressive_moe.py`  
> **기준일**: 2026-03-09

---

## 심각도 기준

| 등급 | 의미 |
|------|------|
| 🔴 치명 | 런타임 오류 또는 학습 결과 무효화 |
| 🟠 높음 | 잠재적 버그 또는 심각한 성능 저하 |
| 🟡 중간 | 비효율 또는 코드 품질 문제 |
| 🟢 낮음 | 스타일·재현성·유지보수 개선 |

---

## 1. model.py

### 🔴 [M-1] 테스트 코드 — 튜플 반환 미처리

**위치**: `__main__` 블록

`TransformerMoE.forward`는 `TransformerBlock`을 레이어로 사용하며, `TransformerBlock._forward_impl`은 `(out, aux_loss)` 튜플을 반환한다.  
현재 테스트 코드는 반환값을 그대로 `.shape` 접근하므로 `AttributeError: 'tuple' object has no attribute 'shape'`가 발생한다.

```python
# ❌ 현재 (버그)
output = model(dummy_input)
print(f"🧪 출력: {output.shape}")

# ✅ 수정
result = model(dummy_input)
if isinstance(result, tuple):
    output, aux_loss = result
    print(f"🧪 출력: {output.shape}, aux_loss: {aux_loss.item():.6f}")
else:
    output = result
    print(f"🧪 출력: {output.shape}")
print("✅ Forward Pass 성공!")
```

---

### 🟠 [M-2] `expert_capacity_factor` — 선언만 있고 로직 없음

**위치**: `ModelConfig`

`expert_capacity_factor: float = 1.25`가 선언되어 있지만 `MoELayer`, `MoEBlock` 어느 곳에서도 참조하지 않는다.  
토큰 드롭(token dropping) 없이 모든 토큰이 그냥 처리되므로 실제로 capacity 제한이 작동하지 않는다.

```python
# ✅ 미구현 명시 또는 구현
# TODO: expert_capacity_factor 미사용.
# capacity = int((tokens_per_expert_ideal) * expert_capacity_factor)
# 초과 토큰은 overflow 버퍼로 라우팅하거나 드롭 처리 예정
expert_capacity_factor: float = 1.25
```

---

### 🟡 [M-3] `top_k_gating` 함수 — 데드 코드

**위치**: `MoELayer` 클래스 위

`top_k_gating` 함수가 정의되어 있지만 `MoELayer.forward`와 `MoEBlock._forward_impl` 모두 `torch.topk + F.softmax`를 직접 인라인으로 사용한다.  
→ 함수를 삭제하거나 내부 로직을 이 함수로 통일.

---

### 🟡 [M-4] `_load_balancing_loss` — `for` 루프 → 벡터화 가능

**위치**: `MoELayer._load_balancing_loss`

`for k_idx in range(num_experts_per_tok):` 루프로 `scatter_add_`를 반복 호출하는데, `view(-1)` 한 번으로 처리 가능하다.

```python
# ✅ 개선
def _load_balancing_loss(self, router_logits, top_k_indices):
    num_experts = self.config.num_experts
    routing_probs = F.softmax(router_logits, dim=-1)
    avg_prob = routing_probs.mean(dim=[0, 1])          # (E,)

    # ← 루프 제거: view(-1) 한 번으로 처리
    flat_indices = top_k_indices.view(-1)              # (bsz*seq*k,)
    expert_counts = torch.zeros(num_experts, device=router_logits.device)
    expert_counts.scatter_add_(
        0, flat_indices,
        torch.ones(flat_indices.numel(), device=router_logits.device)
    )
    fraction = expert_counts / flat_indices.numel()
    return num_experts * (fraction * avg_prob).sum()
```

---

## 2. train_master.py

### 🔴 [T1-1] `total_estimated` — CUDA 없을 때 `NameError`

**위치**: `main()`, 메모리 체크 분기 이후

CUDA GPU가 없는 환경에서 `if total_estimated > 24.0:` 라인에서 `NameError`가 발생한다.

```python
# ✅ 수정: 기본값을 먼저 설정
total_estimated = 0.0
if torch.cuda.is_available():
    # ... (기존 계산 로직 그대로) ...
    total_estimated = model_memory_total + optimizer_memory + activation_memory
```

---

### 🟠 [T1-2] Learning Rate Scheduler 없음

**위치**: `main()`, optimizer 설정 이후

고정 학습률만 사용하고 있다. LLM pretraining에서 **warmup + cosine decay**는 사실상 표준이며, 없을 경우 초반 불안정 및 후반 과적합이 발생하기 쉽다.

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

total_steps = len(dataloader) * num_epochs
warmup_steps = max(1, total_steps // 10)  # 전체의 10%

warmup_sched  = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                         total_iters=warmup_steps)
cosine_sched  = CosineAnnealingLR(optimizer,
                                  T_max=total_steps - warmup_steps, eta_min=1e-6)
scheduler = SequentialLR(optimizer,
                         schedulers=[warmup_sched, cosine_sched],
                         milestones=[warmup_steps])

# 학습 루프 내 optimizer.step() 바로 다음에:
scheduler.step()
```

---

### 🟡 [T1-3] QLoRA Linear4bit 교체 코드 — 7회 반복

**위치**: `main()` 내 QLoRA 적용 블록

`wq`, `wk`, `wv`, `wo`, `w1`, `w2`, `w3` 각각에 대해 동일한 패턴이 7회 반복된다.  
`build_qlora_master_model()` 헬퍼 함수가 **이미 존재**하는데 `main()`에서 이를 사용하지 않고 인라인 중복 구현하고 있다.

```python
# ✅ 수정: 헬퍼 함수 호출로 대체
if use_qlora:
    model = build_qlora_master_model(config, device)
```

---

### 🟢 [T1-4] 재현성 시드 없음

**위치**: `main()` 시작 부분

```python
import random, numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)
```

---

## 3. train_moe.py

### 🔴 [T2-1] Tokenizer 재구축 — Master 임베딩과 ID 불일치

**위치**: `main()`, 데이터 로딩 이후

`train_moe.py`는 Mixed 데이터(나무위키 70% + 코드 30%)로 어휘 사전을 **새로 구축**한다.  
`train_master.py`는 나무위키 단독으로 어휘 사전을 구축했으므로 두 어휘 사전의 토큰 ID가 달라진다.  

**결과**: Master가 학습한 임베딩 가중치가 MoE 모델에서 완전히 다른 의미로 해석된다.  
**Sparse Upcycling의 핵심 전제가 무너지는 치명적 버그다.**

```python
# ✅ 수정: Master 토크나이저를 그대로 로드
tokenizer_path = "master_model_checkpoints/tokenizer.json"
if os.path.exists(tokenizer_path):
    print(f">>> Master 토크나이저 로드: {tokenizer_path}")
    tokenizer = KoreanBPETokenizer.load(tokenizer_path)
else:
    print(">>> ⚠️ 경고: Master 토크나이저 없음 — 임베딩 불일치 가능성!")
    tokenizer = KoreanBPETokenizer(vocab_size=moe_model.config.vocab_size)
    tokenizer.build_vocab(mixed_texts[:10000])
```

---

### 🟠 [T2-2] `gradient_accumulation_steps` 불일치

**위치**: `main()` vs `train_moe_stage()` 내부

```python
# main()에서 설정
gradient_accumulation_steps = 2   # 실효 배치 = 4

# train_moe_stage() 내부 하드코딩 — main() 값이 무시됨
train_moe_model(..., gradient_accumulation_steps=4)   # 실효 배치 = 8
```

`train_moe_stage()` 시그니처에 파라미터를 추가하고 `main()`에서 주입한다.

```python
def train_moe_stage(
    ...,
    gradient_accumulation_steps: int = 4,  # ← 추가
):
    losses = train_moe_model(
        ...,
        gradient_accumulation_steps=gradient_accumulation_steps,  # ← 전달
    )
```

---

### 🟠 [T2-3] Stage 전환 시 Optimizer 재생성 — State 리셋

**위치**: `train_moe_stage()`

Stage 1→2→3마다 optimizer를 새로 생성한다. 각 Stage에서 쌓인 momentum/variance(Adam state)가 초기화되어 Stage 3에서 Router의 학습 이력이 사라진다.

**권고**: 파라미터 그룹별로 `requires_grad`를 on/off하면서 **단일 optimizer를 유지**하거나, Stage 전환 시 기존 state를 복사하는 방식을 사용한다.

```python
# 권고 패턴: 파라미터 그룹 분리 + 단일 optimizer
optimizer = AdamW8bit([
    {"params": router_params,  "lr": 1e-4},
    {"params": expert_params,  "lr": 2e-4},
    {"params": attn_params,    "lr": 0.0},   # Attention은 lr=0으로 사실상 동결
], weight_decay=0.01, betas=(0.9, 0.95))
```

---

### 🟡 [T2-4] Learning Rate Scheduler 없음

`train_master.py`와 동일한 문제. 각 Stage마다 warmup + cosine decay 스케줄 적용을 권고한다.

---

## 4. utils.py

### 🔴 [U-1] `DoRALinear.forward` — Base weight 누락

**위치**: `DoRALinear.forward`

DoRA 수식은 `output = m * normalize(W + ΔW)·x` 인데, 현재 구현은 `W·x`(베이스 선형 변환) 자체가 없고 LoRA delta만 반환한다. 즉 **선형 변환이 전혀 수행되지 않는다**.

```python
# ❌ 현재 (버그): W*x 없음
def forward(self, x):
    lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
    lora_output = lora_output * (self.m / (lora_output.norm(dim=-1, keepdim=True) + 1e-9))
    return lora_output

# ✅ 수정: base weight 추가
class DoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=32, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)  # ← 추가
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.m = nn.Parameter(torch.ones(out_features))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        base_out   = x @ self.weight.T                                        # W·x
        lora_delta = self.dropout(x) @ self.lora_A.T @ self.lora_B.T \
                     * (self.alpha / self.rank)                               # ΔW·x
        combined   = base_out + lora_delta
        norm       = combined.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return combined * (self.m / norm)                                     # magnitude 정규화
```

---

### 🔴 [U-2] argparse 인수 이름 버그

**위치**: `__main__` 블록

```python
# ❌ 현재 (버그): 인수명이 공백 3개 — 실제로 참조 불가
parser.add_argument("   ", action="store_true", default=True, help="FlashAttention 사용")

# 이후 args.use_flash_attention 참조 → AttributeError
print(f"  - FlashAttention: {args.use_flash_attention}")
train(use_flash_attention=args.use_flash_attention, ...)
```

```python
# ✅ 수정
parser.add_argument("--use-flash-attention", action="store_true", default=True,
                    help="FlashAttention 사용")
# argparse가 - → _ 자동 변환: args.use_flash_attention 으로 정상 접근
```

---

### 🟠 [U-3] Deprecated API — `torch.cuda.amp`

**위치**: `train()` 함수

```python
# ❌ deprecated (PyTorch 2.x에서 경고, 향후 제거 예정)
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast(dtype=dtype):

# ✅ 수정
scaler = torch.amp.GradScaler("cuda")
with torch.amp.autocast("cuda", dtype=dtype):
```

> `train_master.py`, `train_moe.py`는 이미 새 API 사용 중 — `utils.py`의 `train()` 함수만 수정 필요.

---

### 🟡 [U-4] `GaLoreAdamW._update_projections` — 미구현 스텁

**위치**: `GaLoreAdamW._update_projections`

200 스텝마다 호출되지만 `pass`만 있어 GaLore의 핵심 기능(SVD 기반 투영 업데이트)이 **비활성화** 상태다.

```python
# ✅ 미구현 명시 (현재 pass보다 명확)
def _update_projections(self):
    raise NotImplementedError(
        "GaLoreAdamW._update_projections 미구현.\n"
        "  → pip install galore-torch 사용 권장: "
        "https://github.com/jiaweizzhao/GaLore"
    )
```

---

### 🟡 [U-5] `TextDataset` — `ignore_index` 하드코딩으로 실제 pad_id와 불일치

**위치**: `TextDataset.__getitem__` 및 `train()` 내 criterion 생성

`CrossEntropyLoss(ignore_index=0)` 이 하드코딩되어 있다.  
`KoreanBPETokenizer`에서 실제 `pad_id`를 동적으로 받아야 한다.

```python
# ✅ 수정: 실제 pad_id 사용
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
```

---

## 5. progressive_moe_layer.py

### 🔴 [P-1] `ProgressiveMoELayer.forward` — `aux_loss` 미반환

**위치**: `ProgressiveMoELayer.forward`

`MoELayer.forward`와 `TransformerBlock._forward_impl`은 `(output, aux_loss)` 튜플을 반환한다.  
`ProgressiveMoELayer.forward`는 `output`만 반환하므로, 이를 `TransformerBlock.moe`에 연결하면 **언팩 시 오류가 발생**하고 load balancing이 전혀 작동하지 않는다.

```python
# ✅ 수정: aux_loss 계산 및 튜플 반환
def forward(self, x):
    bsz, seq_len, dim = x.shape
    router_logits = self.router(x)      # (bsz, seq_len, max_experts)
    k = min(self.num_experts_per_tok, self.current_num_experts)
    top_k_logits, top_k_indices = torch.topk(router_logits, k, dim=-1)
    top_k_gates = F.softmax(top_k_logits, dim=-1)

    # ... (기존 dispatch 로직 그대로) ...

    output = flat_output.view(bsz, seq_len, dim)

    # ← 추가
    aux_loss = self._load_balancing_loss(
        router_logits[:, :, :self.current_num_experts], top_k_indices
    )
    return output, aux_loss

def _load_balancing_loss(self, router_logits, top_k_indices):
    """Switch Transformer style load balancing loss"""
    E = self.current_num_experts
    routing_probs = F.softmax(router_logits, dim=-1)
    avg_prob = routing_probs.mean(dim=[0, 1])          # (E,)
    flat_indices = top_k_indices.view(-1)
    expert_counts = torch.zeros(E, device=router_logits.device)
    expert_counts.scatter_add_(
        0, flat_indices,
        torch.ones(flat_indices.numel(), device=router_logits.device)
    )
    fraction = expert_counts / flat_indices.numel()
    return E * (fraction * avg_prob).sum()
```

---

### 🔴 [P-2] `ExpandableRouter` 초기화 — `config.num_experts`와 `initial_num_experts` 충돌

**위치**: `ProgressiveMoELayer.__init__`

`ExpandableRouter`는 `config.num_experts`를 current로 설정하고  
`expert_mask[config.num_experts:] = False`로 마스킹한다.  

그러나 `ProgressiveMoELayer.__init__`에서 `initial_num_experts=2`, `config.num_experts=8`이면  
Router는 이미 8개가 활성화된 상태로 시작하고, 이후 `add_experts(2)` 호출이 의미 없어진다.

```python
# ✅ 수정: config를 복사해서 num_experts를 initial로 override
import copy

def __init__(self, config: ModelConfig, progressive_config: ProgressiveMoEConfig):
    super().__init__()
    self.config = config
    self.progressive_config = progressive_config

    # config.num_experts를 initial_num_experts로 덮어쓴 복사본으로 Router 초기화
    init_config = copy.copy(config)
    init_config.num_experts = progressive_config.initial_num_experts

    self.router = ExpandableRouter(
        init_config,
        max_experts=progressive_config.max_num_experts
    )
    self.experts = nn.ModuleList()
    self.current_num_experts = 0
    self.add_experts(progressive_config.initial_num_experts)
    self.num_experts_per_tok = config.num_experts_per_tok
```

---

### 🟡 [P-3] `get_expert_usage_stats` — Router weight norm은 실제 사용량과 무관

**위치**: `ProgressiveMoELayer.get_expert_usage_stats`

```python
# ❌ 현재: Router weight의 L2 norm → 실제 라우팅 빈도와 무관
usage_scores = router_weights.norm(dim=1)
```

이 값으로 "best expert"를 선택하면 잘못된 Expert가 복사될 수 있다.

```python
# ✅ 수정: Forward pass에서 실제 라우팅 카운트 누적

# __init__에 추가
self.register_buffer(
    'routing_counts',
    torch.zeros(progressive_config.max_num_experts)
)

# forward() 내부, top_k_indices 계산 직후 추가
with torch.no_grad():
    flat = top_k_indices.view(-1)
    ones = torch.ones(flat.numel(), device=flat.device)
    self.routing_counts[:self.current_num_experts].scatter_add_(0, flat, ones)

# 통계 반환
def get_expert_usage_stats(self):
    counts = self.routing_counts[:self.current_num_experts].clone()
    total  = counts.sum().clamp(min=1.0)
    return (counts / total).cpu().numpy()   # 비율로 정규화
```

---

### 🟡 [P-4] `add_experts` — Router 확장 시 `init_method` 항상 `"zero"` 강제

**위치**: `ProgressiveMoELayer.add_experts`

```python
# 현재: 호출자가 "copy_best"를 요청해도 Router는 항상 "zero"로 초기화
self.router.add_experts(num_new_experts, init_method="zero")  # ← 하드코딩
```

새 Expert를 `"copy_best"`로 초기화했을 때 Router도 해당 Expert에 맞게 bias를 주는 것이 학습 초반 수렴에 유리하다. 최소한 파라미터를 외부에서 받도록 변경한다.

```python
router_init = "zero" if init_method == "random" else "copy_existing"
self.router.add_experts(num_new_experts, init_method=router_init)
```

---

## 6. test_progressive_moe.py

### 🟠 [TS-1] Forward pass 반환값 미처리 — 수정 후 즉시 실패

**위치**: 테스트 전체

`[P-1]` 수정 후 `ProgressiveMoELayer.forward`가 `(output, aux_loss)` 튜플을 반환하면, 현재 테스트 코드는 `output1.shape` 접근에서 즉시 실패한다.

```python
# ❌ 현재
output1 = moe_layer(x)
print(f"   -> 초기 출력: {output1.shape}")

# ✅ 수정
output1, aux_loss1 = moe_layer(x)
print(f"   -> 초기 출력: {output1.shape}, aux_loss: {aux_loss1.item():.6f}")

output2, aux_loss2 = moe_layer(x)
print(f"   -> 확장 후 출력: {output2.shape}, aux_loss: {aux_loss2.item():.6f}")
```

---

### 🟡 [TS-2] 테스트 범위 부족 — 주요 경로 미검증

현재 테스트는 shape 확인만 한다. 수정된 코드에서 검증해야 할 항목들:

```python
# 추가 검증 항목
# 1. Router 초기화 충돌 검증
assert moe_layer.router.current_num_experts == progressive_config.initial_num_experts, \
    "Router current_num_experts 초기화 불일치"

# 2. Expert 동결/해제 검증
moe_layer.freeze_experts([0])
assert not moe_layer.experts[0].w1.weight.requires_grad, "freeze_experts 실패"
moe_layer.unfreeze_experts([0])
assert moe_layer.experts[0].w1.weight.requires_grad, "unfreeze_experts 실패"

# 3. max_num_experts 초과 방지
try:
    moe_layer.add_experts(999)   # max 초과 시도
    assert False, "ValueError 미발생"
except ValueError:
    print("   -> ✅ max_num_experts 초과 방지 확인")

# 4. aux_loss gradient 흐름 확인
output, aux_loss = moe_layer(x)
loss = output.sum() + 0.01 * aux_loss
loss.backward()
for p in moe_layer.router.parameters():
    assert p.grad is not None, "Router gradient 없음"
print("   -> ✅ gradient 흐름 확인")
```

---

## 7. 전체 공통 이슈

### 🟠 [G-1] MoE dispatch 중첩 for 루프 — 3곳 공통 병목

**위치**: `MoELayer.forward`, `ProgressiveMoELayer.forward`, `MoEBlock._forward_impl`

세 곳 모두 동일한 `O(k × E)` 중첩 루프를 사용한다. E=8, k=2 기준으로 루프 16회 반복 + 매번 Boolean masking은 GPU 활용률을 낮춘다.

```python
# ✅ 개선: Expert별 단일 루프 (k 루프 제거)
# 중첩 루프를 Expert 루프 하나로 통합 — 세 곳에 공통 적용 가능

flat_x = x.view(-1, dim)                               # (T, dim)  T = bsz*seq_len
router_logits = self.router(x).view(-1, num_experts)   # (T, E)
top_k_logits, top_k_indices = torch.topk(router_logits, k, dim=-1)
top_k_gates = F.softmax(top_k_logits, dim=-1)          # (T, k)
flat_output = torch.zeros_like(flat_x)

for expert_idx in range(num_experts):
    # 이 expert가 선택된 위치: (T, k) bool
    is_selected = (top_k_indices == expert_idx)
    if not is_selected.any():
        continue

    token_mask = is_selected.any(dim=-1)               # (T,) bool
    # 여러 slot에서 선택된 경우 gate 합산 (드문 경우)
    gate_vals  = (top_k_gates * is_selected.float()).sum(dim=-1)  # (T,)

    expert_out = self.experts[expert_idx](flat_x[token_mask])
    flat_output[token_mask] += expert_out * gate_vals[token_mask].unsqueeze(-1)
```

---

### 🟠 [G-2] Learning Rate Scheduler 전면 미적용

`train_master.py`, `train_moe.py`, `utils.py::train()` 모두 고정 학습률만 사용.

권고 스케줄: **linear warmup (10%) + cosine decay**

```
lr
▲
│  ╱─────╲
│ ╱       ╲─────────────────── eta_min
│╱
└──────────────────────────────▶ steps
 ←10%→←──────── 90% ──────────→
```

---

### 🟢 [G-3] 재현성 시드 전면 미설정

`train_master.py`, `train_moe.py`, `utils.py::train()` 모두 시드 없음 → 실험 재현 불가.

```python
def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

### 🟢 [G-4] TRAINING_GUIDE.md / README — 클래스명 불일치

문서와 실제 코드 사이 혼란을 줄이기 위해 문서를 실제 클래스명으로 업데이트 권고.

| 문서 내 표기 | 실제 코드 | 비고 |
|-------------|----------|------|
| `MoEBlock` | `TransformerBlock` | `MoEBlock = TransformerBlock` 별칭 존재 |
| `JambaConfig` | `ModelConfig` | `JambaConfig = ModelConfig` 별칭 존재 |
| `Jamba2` | `TransformerMoE` | `Jamba2 = TransformerMoE` 별칭 존재 |
| `jamba2_model.py` | `model.py` | README의 파일 구조와 실제 파일명 불일치 |

---

## 수정 우선순위 요약

| 우선순위 | 등급 | 파일 | 항목 | 예상 소요 |
|---------|------|------|------|---------|
| 1 | 🔴 | `train_moe.py` | [T2-1] Tokenizer 재구축 불일치 | 10분 |
| 2 | 🔴 | `progressive_moe_layer.py` | [P-2] Router 초기화 충돌 | 20분 |
| 3 | 🔴 | `progressive_moe_layer.py` | [P-1] aux_loss 미반환 | 20분 |
| 4 | 🔴 | `utils.py` | [U-1] DoRALinear base weight 누락 | 30분 |
| 5 | 🔴 | `utils.py` | [U-2] argparse 이름 버그 | 5분 |
| 6 | 🔴 | `train_master.py` | [T1-1] total_estimated 미정의 | 5분 |
| 7 | 🔴 | `model.py` | [M-1] 테스트 코드 튜플 처리 | 5분 |
| 8 | 🟠 | `test_progressive_moe.py` | [TS-1] forward 반환값 미처리 | 10분 |
| 9 | 🟠 | `train_moe.py` | [T2-2] gradient_accum 불일치 | 10분 |
| 10 | 🟠 | `utils.py` | [U-3] Deprecated AMP API | 5분 |
| 11 | 🟠 | 전체 | [G-1] MoE dispatch 루프 최적화 | 1시간 |
| 12 | 🟠 | 전체 | [G-2] LR Scheduler 추가 | 1시간 |
| 13 | 🟠 | `train_moe.py` | [T2-3] Stage 전환 시 optimizer state 리셋 | 1시간 |
| 14 | 🟡 | `progressive_moe_layer.py` | [P-3] 사용량 통계 정확도 | 30분 |
| 15 | 🟢 | 전체 | [G-3] 재현성 시드 설정 | 15분 |

---

## SAFETY_AI_ROADMAP 구현 시 추가 고려사항

현재 코드 구조를 **안전수칙 위반 분류기**로 전환할 때 아래 항목이 추가로 필요하다.

### 분류 헤드 추가

현재 `output` 레이어는 다음 토큰 예측용(vocab_size 차원). 분류 태스크용으로는 `[CLS]` 토큰 기반 분류 헤드가 필요하다.

```python
class SafetyClassificationHead(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.dense     = nn.Linear(dim, dim)
        self.dropout   = nn.Dropout(0.1)
        self.out_proj  = nn.Linear(dim, num_classes)

    def forward(self, hidden_states):
        # hidden_states[:, 0, :] = [CLS] 토큰 위치
        x = self.dropout(hidden_states[:, 0, :])
        x = F.gelu(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)
```

### 불균형 데이터 처리

SIF 데이터에서 위반 유형별 샘플 수 차이가 크므로:

```python
from torch.utils.data import WeightedRandomSampler

class_counts  = [count_per_class]      # 위반유형별 샘플 수
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = class_weights[labels]  # 각 샘플의 가중치
sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset))
```

또는 **Focal Loss** 사용:

```python
def focal_loss(logits, targets, gamma=2.0):
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    return ((1 - pt) ** gamma * ce_loss).mean()
```

### Expert 역할 유도 (Supervised Routing)

위반 유형별 라벨을 Router auxiliary loss에 반영하면 각 Expert가 특정 위반 유형을 전담하도록 유도할 수 있다.

```python
# aux_loss에 supervised 성분 추가
def supervised_routing_loss(router_logits, violation_type_labels, num_experts):
    """
    violation_type_labels: (bsz, seq_len) — 각 토큰의 위반 유형 ID
    목적: 위반유형 i → Expert i 로 라우팅 유도
    """
    routing_probs = F.softmax(router_logits, dim=-1)   # (bsz, seq_len, E)
    target_expert = violation_type_labels % num_experts  # 유형 → expert 매핑
    target_one_hot = F.one_hot(target_expert, num_classes=num_experts).float()
    return F.kl_div(routing_probs.log(), target_one_hot, reduction='batchmean')
```

### 평가 지표 추가

Loss 외에 위반 유형별 성능 트래킹:

```python
from sklearn.metrics import f1_score, classification_report

# 학습 루프에서
all_preds, all_labels = [], []
# ... 배치마다 append ...
f1 = f1_score(all_labels, all_preds, average='macro')
print(classification_report(all_labels, all_preds, target_names=violation_type_names))
```
