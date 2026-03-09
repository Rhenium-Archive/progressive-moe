# Progressive MoE 코드 리뷰 — 버그·성능·품질 개선 제안

> 분석 대상: `model.py`, `progressive_moe_layer.py`, `train_master.py`, `train_moe.py`, `utils.py`, `test_progressive_moe.py`

---

## 1. 버그 / 정확성 문제 (Bug & Correctness)

### 1-1. `train_moe.py` — `convert_master_to_moe`에서 원본 config 변조
```python
# 현재 (문제)
config = master_model.config
config.num_experts = num_experts   # ← 원본 config 오염!

# 수정
import copy
config = copy.copy(master_model.config)
config.num_experts = num_experts
```
`master_model.config`는 참조(reference)이므로 인플레이스로 수정하면 이후 `master_model`을 재사용할 때 예상치 못한 오류가 발생합니다.

---

### 1-2. `train_master.py` / `train_moe.py` — 마지막 배치 gradient 누락
```python
# 현재: 마지막 배치가 gradient_accumulation_steps의 배수가 아니면 step() 미호출
for batch_idx, ... in enumerate(dataloader):
    ...
    if (batch_idx + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 수정: 루프 종료 후 잔여 gradient 처리
# for 루프 이후 아래 코드 추가:
if (len(dataloader) % gradient_accumulation_steps) != 0:
    optimizer.step()
    optimizer.zero_grad()
```
데이터셋 크기가 `gradient_accumulation_steps`의 배수가 아닐 경우, 마지막 미니배치들의 gradient가 반영되지 않고 버려집니다.

---

### 1-3. `progressive_moe_layer.py` — `_load_balancing_loss`에서 인덱스 범위 초과 가능
```python
# 현재: top_k_indices는 max_experts 범위의 인덱스를 가질 수 있음
active_logits = router_logits[:, :, :self.current_num_experts]
aux_loss = self._load_balancing_loss(active_logits, top_k_indices)
# → _load_balancing_loss에서 scatter_add_(0, flat_indices, ...) 호출 시
#   flat_indices의 최대값이 current_num_experts를 초과하면 런타임 에러

# 수정: top_k_indices도 clamp 처리
flat_indices = top_k_indices.view(-1).clamp(max=self.current_num_experts - 1)
```
`ExpandableRouter`는 최대 `max_experts`까지 인덱스를 반환할 수 있으나, `aux_loss` 계산 시 `E = current_num_experts` 크기의 텐서에 scatter하여 out-of-bound가 발생할 수 있습니다.

---

### 1-4. `progressive_moe_layer.py` — `add_experts` 원자성 부재
```python
# 현재: experts 추가 도중 예외 발생 시 router와 experts 간 불일치 상태
for i in range(num_new_experts):
    self.experts.append(new_expert)        # 일부만 추가될 수 있음
    self.current_num_experts += 1

self.router.add_experts(num_new_experts, ...)  # router만 확장

# 수정: 전체 experts 생성 후 한꺼번에 추가
new_experts = [self._create_expert(init_method) for _ in range(num_new_experts)]
self.router.add_experts(num_new_experts, init_method=router_init)  # router 먼저
for e in new_experts:
    self.experts.append(e)
self.current_num_experts += num_new_experts
```

---

### 1-5. `utils.py` — `KoreanBPETokenizer.load`에서 `__init__` 미호출
```python
# 현재: __new__로 생성해 enable_padding 등 초기화 누락
@classmethod
def load(cls, path):
    instance = cls.__new__(cls)
    instance.tokenizer = Tokenizer.from_file(path)
    ...

# 수정: 최소한 padding 설정 재적용
@classmethod
def load(cls, path):
    instance = cls.__new__(cls)
    instance.tokenizer = Tokenizer.from_file(path)
    instance.vocab_size = instance.tokenizer.get_vocab_size()
    instance.unk_id = instance.tokenizer.token_to_id("<unk>")
    instance.pad_id = instance.tokenizer.token_to_id("<pad>") or 0
    instance.eos_id = instance.tokenizer.token_to_id("</s>")
    instance.bos_id = instance.tokenizer.token_to_id("<s>")
    # padding 재설정
    instance.tokenizer.enable_padding(
        pad_id=instance.pad_id, pad_token="<pad>", length=None
    )
    return instance
```

---

### 1-6. `train_moe.py` — `target_total` ZeroDivisionError 가능성
```python
# 현재
target_total = min(len(namu_texts) // namu_ratio, len(code_texts) // code_ratio)
# namu_ratio = 0.7 → //는 float에서 정수가 아니므로 의도와 다름 (float division)

# 수정
target_total = int(min(len(namu_texts) / namu_ratio, len(code_texts) / code_ratio))
target_total = min(target_total, 50000)
```

---

## 2. 성능 개선 (Performance)

### 2-1. `model.py` / `progressive_moe_layer.py` — MoE forward 중첩 루프 → 벡터화

현재 구현은 `O(k × E)` 중첩 루프를 사용합니다. expert 수가 늘어날수록 오버헤드가 커집니다.

```python
# 현재 (O(k × E) Python 루프)
for k_idx in range(k):
    selected_experts = flat_top_k_indices[:, k_idx]
    for expert_idx in range(self.current_num_experts):
        expert_mask = (selected_experts == expert_idx)
        if not expert_mask.any():
            continue
        ...

# 개선: einsum 기반 배치 dispatch (scatter/gather 활용)
# flat_x: (T, D), flat_top_k_indices: (T, k), flat_top_k_gates: (T, k)
T, D = flat_x.shape
# 각 (토큰, k) 쌍을 (T*k, D)로 펼침
tokens_for_dispatch = flat_x.unsqueeze(1).expand(-1, k, -1).reshape(T * k, D)
expert_ids = flat_top_k_indices.reshape(T * k)        # (T*k,)
weights    = flat_top_k_gates.reshape(T * k, 1)       # (T*k, 1)

flat_output = torch.zeros_like(flat_x)
for expert_idx in range(self.current_num_experts):
    mask = (expert_ids == expert_idx)                 # 한 루프만
    if not mask.any():
        continue
    out = self.experts[expert_idx](tokens_for_dispatch[mask])
    # scatter_add로 원래 위치에 가중합
    token_indices = torch.where(mask)[0] % T          # (T*k → T)로 역매핑
    flat_output.scatter_add_(
        0,
        token_indices.unsqueeze(1).expand(-1, D),
        out * weights[mask]
    )
```
단, expert 수가 충분히 많을 때는 [Megablocks](https://github.com/stanford-futuredata/megablocks) 같은 라이브러리 도입이 더 효율적입니다.

---

### 2-2. `model.py` — `_load_balancing_loss`에서 `F.softmax` 이중 호출 제거

```python
# 현재: forward()에서 softmax → aux_loss에서 또 softmax
top_k_gates = F.softmax(top_k_logits, dim=-1)     # forward에서
...
routing_probs = F.softmax(router_logits, dim=-1)  # aux_loss에서 전체 재계산

# 개선: routing_probs를 forward에서 한 번만 계산해 전달
routing_probs = F.softmax(router_logits, dim=-1)
top_k_gates   = routing_probs.topk(k, dim=-1)     # softmax 결과로 top-k 추출
# → 단, top-k after softmax는 top-k logits after softmax와 동일하므로 OK
```

---

### 2-3. `progressive_moe_layer.py` — `routing_counts` 업데이트 시 임시 텐서 절감

```python
# 현재
ones = torch.ones(flat.numel(), device=flat.device)
self.routing_counts[:self.current_num_experts].scatter_add_(0, flat, ones)

# 개선: ones_like 대신 미리 고정 텐서 재사용, 또는
self.routing_counts.scatter_add_(0, flat, flat.new_ones(flat.numel()))
```

---

### 2-4. `model.py` — 가중치 초기화 개선 (depth-scaled init)

GPT-2 논문 방식: residual projection의 std를 레이어 깊이에 따라 조정하면 깊은 모델에서 학습이 안정됩니다.

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        std = 0.02
        # residual projection(wo, w2)에는 깊이 보정
        if hasattr(module, '_is_residual'):
            std = 0.02 / math.sqrt(2 * self.config.n_layers)
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
```

---

### 2-5. `train_master.py` — DataLoader `num_workers=0` 고정 → 병렬 로딩 활성화

```python
# 현재: num_workers=0 (단일 스레드)
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, ...)

# 개선
import multiprocessing
num_workers = 0 if os.name == 'nt' else min(4, multiprocessing.cpu_count() // 2)
dataloader = DataLoader(
    dataset, batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
    pin_memory=True,
)
```

---

### 2-6. `model.py` — `Attention.forward` 추론 시 FlashAttention 미사용

```python
# 현재: 학습 중에만 FlashAttention 사용
if self.use_flash_attention and self.training:
    output = self.flash_attn_func(...)

# 개선: 추론에도 사용 (FlashAttention v2는 inference 지원)
if self.use_flash_attention:
    output = self.flash_attn_func(
        xq, xk, xv,
        dropout_p=self.dropout if self.training else 0.0,
        causal=True
    )
```

---

### 2-7. `model.py` — `freqs_cis` 디바이스 분리 사전 계산

```python
# 현재: CPU에서 precompute하고 forward마다 .to(h.device) 호출
self.freqs_cis = precompute_freqs_cis(config.head_dim, config.max_seq_len * 2)
...
freqs_cis = self.freqs_cis[:seqlen].to(h.device)  # 매 forward마다 device 이동

# 개선: register_buffer로 등록하면 .to(device), .state_dict() 자동 연동
self.register_buffer(
    'freqs_cis',
    precompute_freqs_cis(config.head_dim, config.max_seq_len * 2),
    persistent=False
)
```

---

## 3. 코드 품질 / 리팩터링 (Code Quality)

### 3-1. `train_master.py` — QLoRA 레이어 교체 루프 중복 제거

`main()`에서 각 레이어마다 7개의 Linear를 수동으로 교체하는 코드가 약 200줄 반복됩니다. 이미 `build_qlora_master_model` 헬퍼 함수가 존재하므로 `main()`에서도 이를 사용하면 됩니다.

```python
# 현재: main()에서 직접 for layer in model.layers: ... (200줄)
# 개선: 이미 구현된 헬퍼 함수 사용
if use_qlora:
    model = build_qlora_master_model(config, device=device)
```

---

### 3-2. `utils.py` — `GaLoreAdamW`와 `apply_dora_to_model`의 미완성 구현 명확화

`GaLoreAdamW._update_projections`는 `NotImplementedError`를 발생시키고, `apply_dora_to_model`은 모델을 아무 변경 없이 그대로 반환합니다. 이 두 함수는 실제로 사용 시 오류가 발생하거나 효과가 없어 혼란을 줍니다.

```python
# 개선안 A: 미구현 클래스/함수를 파일 최상단에 명시적으로 표시
# WARNING: GaLoreAdamW is a stub. Install galore-torch for actual implementation.
# pip install galore-torch

# 개선안 B: 가드 추가
def apply_dora_to_model(model, ...):
    raise NotImplementedError(
        "DoRA 적용이 구현되지 않았습니다. "
        "utils.py의 DoRALinear를 각 레이어에 직접 교체해야 합니다."
    )
```

---

### 3-3. `model.py` — `expert_capacity_factor` TODO 처리

현재 `expert_capacity_factor = 1.25`가 `ModelConfig`에 정의되어 있으나 어디서도 사용되지 않습니다. 실제 토큰 드롭(token dropping)을 구현하지 않을 것이라면 필드를 제거하거나 아래처럼 capacity-constrained dispatch를 추가하는 것을 권장합니다.

```python
# capacity = int(expert_capacity_factor * tokens_per_expert)
# tokens_per_expert = (bsz * seq_len * num_experts_per_tok) / num_experts
# expert_input을 capacity 이상으로 받으면 초과 토큰 드롭
```

---

### 3-4. `train_moe.py` — `train_moe_stage`에서 freeze 로직 일관성

`train_moe_stage` 내에서 항상 `moe_model.freeze_attention()`을 호출한 뒤 router/expert freeze를 선택적으로 적용합니다. 이는 Attention이 항상 동결된다는 사실이 함수 시그니처에 드러나지 않아 오해를 유발합니다.

```python
# 개선: 파라미터로 명시
def train_moe_stage(
    ...,
    freeze_attention: bool = True,   # 명시적 파라미터 추가
):
```

---

### 3-5. `model.py` — `TransformerMoE.forward`의 aux_loss 반환 비일관성

`MasterBlock`은 `h`만 반환하고, `TransformerBlock`은 `(h, aux_loss)` 튜플을 반환합니다. 이를 `isinstance(result, tuple)` 체크로 처리하지만, 타입이 불일관해 유지보수가 어렵습니다.

```python
# 개선: 항상 (output, aux_loss) 형태로 통일, aux_loss가 없으면 0.0 반환
class MasterBlock(nn.Module):
    def forward(self, x, freqs_cis):
        ...
        return out, torch.tensor(0.0, device=x.device)  # aux_loss = 0으로 통일
```

---

### 3-6. `train_master.py` — `torch.load(..., weights_only=False)` 보안 경고

```python
# 현재
ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

# 개선: 신뢰하는 체크포인트라면 weights_only=True로 바꾸거나
# 최소한 경고 메시지 추가
import warnings
warnings.warn(
    f"weights_only=False로 {checkpoint_path}를 로드합니다. "
    "신뢰할 수 없는 파일이라면 보안 위험이 있습니다.",
    UserWarning
)
```

---

## 4. 테스트 코드 개선 (`test_progressive_moe.py`)

| 항목 | 현재 상태 | 개선 제안 |
|------|-----------|-----------|
| 테스트 프레임워크 | 단순 assert + print | `pytest`로 전환해 CI 연동 |
| 엣지 케이스 | `max_num_experts` 초과만 테스트 | `current_num_experts=1`, `num_experts_per_tok > current` 등 추가 |
| 디바이스 커버리지 | CPU만 | CUDA 사용 가능 시 GPU 테스트 자동 실행 |
| aux_loss 범위 검증 | 없음 | `assert 0 <= aux_loss.item()` 추가 |
| routing_counts 검증 | 없음 | forward 후 `routing_counts.sum() > 0` 검증 |

```python
# pytest 전환 예시
import pytest

def test_router_init_consistency():
    ...
    assert moe_layer.router.current_num_experts == progressive_config.initial_num_experts

def test_aux_loss_nonnegative():
    ...
    _, aux_loss = moe_layer(x)
    assert aux_loss.item() >= 0.0

def test_max_experts_guard():
    with pytest.raises(ValueError):
        small_layer.add_experts(999)
```

---

## 5. 요약 우선순위

| 우선순위 | 항목 | 파일 | 영향 |
|----------|------|------|------|
| 🔴 높음 | 1-1. config 원본 변조 | `train_moe.py` | 재현 불가 버그 |
| 🔴 높음 | 1-2. 마지막 배치 누락 | `train_master.py`, `train_moe.py` | 학습 품질 저하 |
| 🔴 높음 | 1-3. scatter 인덱스 범위 초과 | `progressive_moe_layer.py` | 런타임 에러 |
| 🟠 중간 | 1-5. `KoreanBPETokenizer.load` 초기화 누락 | `utils.py` | padding 미적용 |
| 🟠 중간 | 2-1. MoE forward 벡터화 | `model.py`, `progressive_moe_layer.py` | 학습 속도 향상 |
| 🟠 중간 | 2-7. `freqs_cis` register_buffer | `model.py` | 디바이스 이동 오버헤드 |
| 🟡 낮음 | 3-1. QLoRA 루프 중복 | `train_master.py` | 유지보수성 |
| 🟡 낮음 | 3-2. 미완성 stub 명확화 | `utils.py` | 사용자 혼란 방지 |
| 🟡 낮음 | 3-5. aux_loss 반환 통일 | `model.py` | 타입 안전성 |
