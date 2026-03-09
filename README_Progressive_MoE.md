# 점진적 MoE (Progressive Mixture of Experts) 구현

이 프로젝트는 Jamba2 모델에서 **점진적 MoE 학습**을 구현합니다. Main MoE를 먼저 학습하고, 나중에 추가 Expert들을 동적으로 확장할 수 있습니다.

## 🎯 주요 특징

- **동적 Expert 확장**: 학습 중 또는 학습 후 Expert 수를 점진적으로 증가
- **Router 자동 확장**: 새로운 Expert에 맞춰 Router 가중치 자동 확장
- **기존 모델 호환**: 이미 학습된 모델을 Progressive MoE로 변환
- **메모리 효율적**: 단계별 학습으로 메모리 사용량 최적화
- **다양한 초기화 전략**: 새 Expert 초기화 방법 선택 가능

## 📁 파일 구조

```
├── jamba2_model.py              # 기본 Jamba2 모델 구현
├── progressive_moe_layer.py      # 점진적 MoE 레이어 구현
├── train_progressive_moe.py      # 점진적 학습 스크립트
├── expert_expansion_utils.py     # 기존 모델 확장 유틸리티
└── README_Progressive_MoE.md    # 이 파일
```

## 🚀 사용법

### 1. 처음부터 점진적 학습

```bash
# 2개 Expert → 4개 → 6개로 점진적 확장
python train_progressive_moe.py
```

#### 학습 과정:
1. **Stage 1**: 2개 Expert로 기본 MoE 학습
2. **Stage 2**: 4개로 확장 (새 Expert 2개 추가)
3. **Stage 3**: 6개로 최종 확장 (새 Expert 2개 추가)

### 2. 기존 모델에 Expert 추가

```python
from expert_expansion_utils import ExpertExpansionPipeline

# 파이프라인 생성
pipeline = ExpertExpansionPipeline("기존_모델.pt")

# 원본 모델 로드
model, config = pipeline.load_original_model()

# Progressive 설정
progressive_config = pipeline.create_progressive_config(
    target_num_experts=8,      # 목표 Expert 수
    experts_per_stage=2,       # 단계별 추가 수
    new_expert_init="copy_best" # 초기화 방법
)

# 모델 확장
expanded_model = pipeline.convert_and_expand()

# 저장
pipeline.save_expanded_model(expanded_model, "확장된_모델_폴더")
```

### 3. 커스텀 점진적 학습

```python
from progressive_moe_layer import ProgressiveMoELayer, ProgressiveMoEConfig

# 설정
config = JambaConfig(dim=1024, num_experts=2)
progressive_config = ProgressiveMoEConfig(
    base_config=config,
    initial_num_experts=2,
    max_num_experts=8,
    experts_per_stage=2,
    new_expert_init="copy_best"  # "random", "copy_best", "interpolate"
)

# Progressive MoE 레이어 생성
moe_layer = ProgressiveMoELayer(config, progressive_config)

# Expert 추가
moe_layer.add_experts(2, init_method="copy_best")

# 특정 Expert 동결 (기존 지식 보존)
moe_layer.freeze_experts([0, 1])  # 첫 2개 Expert 동결
```

## ⚙️ 설정 옵션

### ProgressiveMoEConfig

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `initial_num_experts` | 시작 Expert 수 | 2 |
| `max_num_experts` | 최대 Expert 수 | 8 |
| `experts_per_stage` | 단계별 추가 수 | 2 |
| `new_expert_init` | 새 Expert 초기화 방법 | "random" |
| `router_expansion` | Router 확장 방식 | "extend" |

### Expert 초기화 방법

1. **"random"**: 랜덤 초기화 (완전히 새로운 Expert)
2. **"copy_best"**: 가장 성능 좋은 기존 Expert 복사 + 노이즈
3. **"interpolate"**: 기존 Expert들의 가중 평균으로 초기화

## 📊 성능 최적화

### 메모리 사용량 비교

| 방법 | Expert 수 | 메모리 사용량 | 학습 시간 |
|------|-----------|---------------|-----------|
| 일반 MoE | 8개 | 높음 | 길음 |
| Progressive MoE | 2→4→8 | 낮음 | 단계적 |

### 권장 설정

**RTX 3090 24GB:**
```python
config = JambaConfig(
    dim=1024,           # 모델 차원
    n_layers=4,         # 레이어 수
    n_heads=8,          # 어텐션 헤드
    max_seq_len=2048    # 시퀀스 길이
)

progressive_config = ProgressiveMoEConfig(
    initial_num_experts=2,
    max_num_experts=6,
    experts_per_stage=2
)
```

**RTX 4090 24GB:**
```python
config = JambaConfig(
    dim=1536,           # 더 큰 모델
    n_layers=6,
    n_heads=12,
    max_seq_len=4096    # 더 긴 컨텍스트
)

progressive_config = ProgressiveMoEConfig(
    initial_num_experts=2,
    max_num_experts=8,
    experts_per_stage=2
)
```

## 🔍 Expert 사용량 분석

```python
# Expert 사용량 통계
usage_stats = moe_layer.get_expert_usage_stats()
print(f"Expert 사용량: {usage_stats}")

# 가장 많이 사용되는 Expert 찾기
best_expert_idx = usage_stats.argmax()
print(f"가장 활성화된 Expert: {best_expert_idx}")
```

## 📈 학습 전략

### 1. Cold Start (처음부터)
```
Stage 1: [Expert A, Expert B] → 기본 패턴 학습
Stage 2: [A, B, C, D] → 특화 패턴 학습  
Stage 3: [A, B, C, D, E, F] → 고급 패턴 학습
```

### 2. Warm Start (기존 모델 확장)
```
기존 모델: [Expert 1, 2, 3, 4] → 이미 학습됨
확장: [1, 2, 3, 4, 5, 6] → 새로운 도메인 학습
```

### 3. 점진적 Fine-tuning
```
1. 기존 Expert 동결
2. 새 Expert만 학습
3. 전체 미세 조정
```

## 🛠️ 고급 사용법

### 동적 Expert 관리

```python
# 런타임 중 Expert 추가
if performance_threshold_reached():
    moe_layer.add_experts(2, "copy_best")
    
# Expert 선택적 동결
moe_layer.freeze_experts([0, 1])  # 기존 지식 보존
moe_layer.unfreeze_experts([2, 3])  # 새 Expert만 학습
```

### 체크포인트 관리

```python
from progressive_moe_layer import ProgressiveMoETrainer

trainer = ProgressiveMoETrainer(model, progressive_config)

# 각 단계별 저장
trainer.expand_model(stage=2)  # Stage 2로 확장
trainer.save_stage_checkpoint("checkpoints", stage=2)

# 특정 단계에서 재시작
trainer.load_stage_checkpoint("checkpoints/progressive_stage_2.pt")
```

## 🚨 주의사항

1. **메모리**: Expert 수가 증가하면 메모리 사용량도 증가
2. **호환성**: 기존 모델 형식과 새 모델 형식이 다를 수 있음
3. **학습률**: 새 Expert 추가 시 더 낮은 학습률 사용 권장
4. **배치 크기**: GPU 메모리에 맞춰 배치 크기 조정 필요

## 🔧 문제 해결

### Q: "RuntimeError: CUDA out of memory"
**A**: 배치 크기를 줄이거나 모델 차원을 축소하세요.

### Q: Router가 새 Expert를 선택하지 않음
**A**: Router 초기화를 "zero"로 설정하고 더 낮은 학습률 사용하세요.

### Q: 기존 모델 성능이 저하됨  
**A**: 기존 Expert를 동결하고 점진적 학습을 사용하세요.

## 📝 라이센스

이 구현은 연구 및 교육 목적으로 제공됩니다.

## 🤝 기여

이슈나 개선 제안은 언제든 환영합니다!