# Sparse Upcycling MoE 학습 가이드

## 개요

단일 FFN을 가진 **Dense 모델(Master)**을 먼저 학습한 뒤, FFN을 N개 Expert로 복제하여 **Sparse MoE 모델**로 확장하는 2단계 파이프라인. Mixtral 방식의 표준 MoE 구조를 따른다.

```
[Phase 1] Dense 모델 학습 (Attention + 단일 FFN)
    ↓ FFN을 Expert로 복제 (Sparse Upcycling)
[Phase 2] MoE 모델 학습 (Attention + Router + N Experts)
```

> **Sparse Upcycling** (Komatsuzaki et al., 2022): 학습된 Dense 모델의 FFN을 복제하여 MoE Expert로 사용하는 기법. 처음부터 MoE를 학습하는 것보다 학습 효율이 높다.

---

## 파일 구조

```
progressive_moe/
├── model.py                      # 모델 아키텍처 정의
│   ├── ModelConfig                #   설정 dataclass
│   ├── Attention                  #   Multi-head attention (RoPE, FlashAttention)
│   ├── Expert                     #   SwiGLU FFN (w1 Gate, w2 Down, w3 Up)
│   ├── Router                     #   Top-k expert 라우팅
│   ├── MasterBlock                #   Attention + 단일 FFN (Phase 1용)
│   ├── MoEBlock                   #   Attention + Router + Experts (Phase 2용, Mixtral 방식)
│   └── TransformerMoE             #   전체 모델 (Attention + MoE)
│
├── utils.py                      # 공용 유틸리티
│   ├── TextDataset               #   텍스트 → 토큰 Dataset
│   ├── KoreanBPETokenizer        #   한국어 BPE 토크나이저
│   ├── get_gpu_memory()          #   GPU 메모리 조회
│   └── print_memory_stats()      #   GPU 메모리 출력
│
├── train_master.py               # [Phase 1] Master (Dense) 모델 학습
├── train_moe.py                  # [Phase 2] MoE 변환 + 단계별 학습
├── progressive_moe_layer.py      # 동적 Expert 확장 레이어 (런타임 확장용)
└── test_progressive_moe.py       # 테스트
```

---

## Phase 1: Master (Dense) 모델 학습

### 실행

```bash
python train_master.py
```

### 목적

MoE 없이 **단일 FFN(Expert)**만으로 기본 언어 패턴을 학습한다.
이 FFN이 이후 모든 Expert의 초기 가중치가 된다.

### 아키텍처 — MasterBlock

```
Input
  ↓
RMSNorm → Attention → + residual
  ↓
RMSNorm → FFN (SwiGLU) → + residual
  ↓
Output
```

- MoE 없음, Router 없음
- 단일 FFN이 모든 토큰을 처리
- 이 FFN 가중치가 Phase 2에서 모든 Expert의 초기값이 됨

### 모델 설정

| 설정 | QLoRA (기본) | Full Fine-tuning |
|------|-------------|-----------------|
| dim | 4096 | 2304 |
| n_layers | 40 | 26 |
| n_heads | 32 | 18 |
| head_dim | 128 | 128 |
| max_seq_len | 2048 | 1920 |
| vocab_size | 32000 | 32000 |
| 예상 파라미터 | ~4-5B | ~1.4-1.8B |
| 예상 VRAM | ~8-12GB | ~20-23GB |

### 학습 설정

| 항목 | QLoRA | Full FT |
|------|-------|---------|
| 양자화 | NF4 4-bit | 없음 |
| LoRA rank / alpha | 16 / 32 | — |
| 학습률 | 2e-4 | 3e-4 |
| 배치 크기 | 4 | 2 |
| Gradient Accumulation | 2 (실질 배치 8) | 2 (실질 배치 4) |
| Optimizer | 8-bit AdamW | 8-bit AdamW |
| Mixed Precision | bfloat16 | bfloat16 |
| Gradient Checkpointing | O | O |

### 데이터

- `heegyu/namuwiki-extracted` (나무위키) 50,000건
- KoreanBPETokenizer로 BPE 어휘 사전 구축 (10,000건 기준)

### 출력 체크포인트

```
master_model_checkpoints/
├── master_model_for_moe.pt    # MoE 변환용 full-precision (QLoRA→디양자화+merge 완료)
├── master_model_epoch_1.pt    # 에폭별 체크포인트
├── training_info.pt           # 학습 정보 (QLoRA 사용 시)
├── adapter_model.bin          # LoRA 어댑터 (QLoRA 사용 시)
└── tokenizer.json             # BPE 토크나이저
```

> `master_model_for_moe.pt`가 Phase 2의 입력이 된다.

---

## Phase 2: MoE 모델 학습

### 실행

```bash
python train_moe.py
```

### 전제 조건

Phase 1에서 생성된 `master_model_checkpoints/master_model_for_moe.pt` 필요.

### 아키텍처 — MoEBlock (Mixtral 방식)

```
Input
  ↓
RMSNorm → Attention → + residual
  ↓
RMSNorm → Router → Top-k Experts → + residual  ← FFN을 완전 대체
  ↓
Output, aux_loss
```

- **Router**: 각 토큰을 Top-k Expert로 라우팅 (학습 가능)
- **Experts**: Master FFN을 복제+노이즈로 초기화, 각자 전문화 (학습 가능)
- **aux_loss**: Load balancing loss (Switch Transformer 방식, Router Collapse 방지)
- 별도의 Master FFN 없음 — Expert들이 FFN을 완전 대체

### Sparse Upcycling 변환

| 항목 | 설정 |
|------|------|
| Expert 수 | 2 (24GB GPU), 4 (32GB+) |
| 활성 Expert/토큰 | 2 (Top-2 라우팅) |
| Expert 초기화 | `copy_with_noise` (Master FFN + Gaussian noise, std의 1%) |

```
변환 과정 (train_moe.py 내부):
1. Master 모델 로드 (master_model_for_moe.pt)
2. MoEModel 생성 (MoEBlock x n_layers)
3. 각 레이어별:
   - Attention 가중치 → 그대로 복사
   - FFN Norm 가중치 → 그대로 복사
   - Master FFN → Expert 0, 1, ... 에 복사 + 노이즈 (Symmetry Breaking)
   - Router → 새로 초기화
4. Embedding + 출력 레이어 → 그대로 복사
```

### 3단계 학습

#### Stage 1: Router 학습

| 항목 | 값 |
|------|------|
| 학습 대상 | Router만 |
| 동결 | Attention + Expert |
| 학습률 | 1e-4 |
| Epochs | 1 |
| 목적 | Router가 토큰을 적절한 Expert로 라우팅하도록 학습 |

#### Stage 2: Expert 학습

| 항목 | 값 |
|------|------|
| 학습 대상 | Expert만 |
| 동결 | Attention + Router |
| 학습률 | 2e-4 |
| Epochs | 1 |
| 목적 | Expert들이 각자의 전문 분야로 분화 |

#### Stage 3: 전체 미세조정

| 항목 | 값 |
|------|------|
| 학습 대상 | Router + Expert |
| 동결 | Attention |
| 학습률 | 1e-4 |
| Epochs | 1 |
| 목적 | Router와 Expert를 함께 최적화 |

> 모든 Stage에서 **Attention은 동결**하여 Phase 1에서 학습된 표현 능력을 보존한다.

### 데이터 — Data Mixing

| 데이터셋 | 비율 | 출처 |
|----------|------|------|
| 나무위키 (한국어) | 70% | `heegyu/namuwiki-extracted` |
| Python 코드 | 30% | `flytech/python-codes-25k` |

- 10개 단위로 7개 나무위키 + 3개 코드를 교차 배치 후 셔플
- 최대 50,000건

### 공통 학습 설정

| 항목 | 값 |
|------|------|
| Optimizer | 8-bit AdamW (betas=0.9, 0.95) |
| Weight Decay | 0.01 |
| Mixed Precision | bfloat16 |
| Gradient Accumulation | 4 |
| Gradient Clipping | max_norm=1.0 |
| Gradient Checkpointing | O |
| 배치 크기 | 2 |
| Load Balancing Loss 계수 | 0.01 |

### Loss 함수

```
total_loss = CrossEntropyLoss(ignore_index=0) + 0.01 * aux_loss
```

- **CrossEntropyLoss**: 다음 토큰 예측 (언어 모델링)
- **aux_loss**: Load balancing loss (균등 분배 시 ≈ 1.0)
  - `aux_loss = N * Σ(f_i × P_i)`
  - f_i: Expert i에 라우팅된 토큰 비율
  - P_i: Expert i의 평균 라우팅 확률

### 출력 체크포인트

```
moe_model_checkpoints/
├── moe_stage1_router.pt       # Stage 1 완료
├── moe_stage2_expert.pt       # Stage 2 완료
├── moe_stage3_full.pt         # Stage 3 완료
├── moe_model_final.pt         # 최종 모델
└── tokenizer.json             # BPE 토크나이저
```

---

## 전체 학습 흐름 요약

```
┌─────────────────────────────────────────────────────┐
│  Phase 1: python train_master.py                    │
│                                                     │
│  Dense Model (Attention + 단일 FFN)                 │
│  ├─ QLoRA 4-bit + LoRA (r=16)                       │
│  ├─ 나무위키 50K건                                   │
│  ├─ lr=2e-4, batch=4, accum=2                       │
│  └─ 출력: master_model_for_moe.pt                   │
└───────────────────────┬─────────────────────────────┘
                        │ FFN → Expert 복사 + 노이즈
                        │ (Sparse Upcycling)
                        ▼
┌─────────────────────────────────────────────────────┐
│  Phase 2: python train_moe.py                       │
│                                                     │
│  MoE Model (Attention + Router + Experts)           │
│  ├─ Mixtral 방식: FFN을 Router+Experts로 완전 대체   │
│  ├─ Attention 동결 (Phase 1 가중치 보존)             │
│  ├─ 나무위키 70% + Python 코드 30%                   │
│  │                                                  │
│  ├─ Stage 1: Router만 학습     (lr=1e-4, 1 epoch)   │
│  ├─ Stage 2: Expert만 학습     (lr=2e-4, 1 epoch)   │
│  ├─ Stage 3: Router+Expert 학습 (lr=1e-4, 1 epoch)  │
│  │                                                  │
│  └─ 출력: moe_model_final.pt                        │
└─────────────────────────────────────────────────────┘
```

---

## GPU 메모리 요구 사항

### Phase 1

| 모드 | 모델 크기 | 예상 VRAM |
|------|----------|----------|
| QLoRA (기본) | ~4-5B | 8-12 GB |
| Full Fine-tuning | ~1.4-1.8B | 20-23 GB |

### Phase 2

| Expert 수 | 추가 파라미터 | 예상 VRAM (Phase 1 QLoRA 기준) |
|-----------|-------------|-------------------------------|
| 2 | FFN x 2 + Router | 16-20 GB |
| 4 | FFN x 4 + Router | 20-24 GB |

> Phase 2는 full-precision으로 실행되므로 Expert 수에 따라 VRAM이 크게 증가한다.
> 24GB GPU에서는 Expert 2개를 권장한다.

---

## 체크포인트 로드 예시

### Master 모델 로드

```python
from train_master import MasterModel, load_master_for_moe

# 방법 1: full-precision 체크포인트 (권장)
import torch
ckpt = torch.load("master_model_checkpoints/master_model_for_moe.pt", map_location="cpu")
model = MasterModel(ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"])

# 방법 2: QLoRA 체크포인트 자동 변환
model = load_master_for_moe("master_model_checkpoints/master_model_epoch_1.pt")
```

### MoE 모델 로드

```python
from train_moe import MoEModel
import torch

ckpt = torch.load("moe_model_checkpoints/moe_model_final.pt", map_location="cpu")
model = MoEModel(ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"])
```

### 토크나이저 로드

```python
from utils import KoreanBPETokenizer

tokenizer = KoreanBPETokenizer.load("moe_model_checkpoints/tokenizer.json")
tokens = tokenizer.encode("안녕하세요")
text = tokenizer.decode(tokens)
```
