# 안전수칙 위반 AI 식별 시스템 로드맵

## 최종 목표

안전수칙 위반 사항을 AI로 식별하는 2단계 시스템.
- **Phase A**: 텍스트 기반 — 재해사례/보고서에서 위반 유형 자동 분류
- **Phase B**: 이미지 기반 — 위반 사진을 입력하면 어떤 수칙을 위반했는지 자동 판별

```
┌──────────────────────────────────────────────────────┐
│  Phase A: 텍스트 MoE 모델                              │
│                                                      │
│  "작업자가 안전대 없이 비계 작업 중 추락"               │
│     ↓ MoE 모델                                       │
│  위반유형: 보호구 미착용 / 법령: 산안법 38조            │
│  위험등급: 높음 / 조치: 작업중지 + 안전대 지급          │
└──────────────────────┬───────────────────────────────┘
                       │ (연동)
┌──────────────────────▼───────────────────────────────┐
│  Phase B: 이미지 → 위반 식별 (추후)                    │
│                                                      │
│  위반 사진 입력 (안전팀 촬영)                           │
│     ↓ 비전 모델 (객체 탐지 + 분류)                     │
│  감지: "안전모 미착용", "안전대 미착용"                  │
│     ↓ Phase A 텍스트 모델에 전달                       │
│  위반유형 + 관련법령 + 대응조치 자동 생성               │
└──────────────────────────────────────────────────────┘
```

---

## Phase A: 텍스트 MoE 모델

### A-1. 데이터 수집

#### 핵심 데이터 — SIF 고위험요인 목록 (최우선)

| 항목 | 내용 |
|------|------|
| 출처 | https://www.data.go.kr/data/15140383/fileData.do |
| 건수 | 4,432건 (제조 1,835건 + 건설 2,574건) |
| 기간 | 2016~2021 사망사고 분석 |
| 컬럼 | 업종(대/중/소), 재해개요, 기인물, 고위험작업/상황, 재해유발요인, 위험성감소대책 |
| 형식 | CSV/엑셀 |
| 장점 | 이미 위반유형별 분류 완료 → 라벨링 불필요 |

#### 보조 데이터

| 데이터 | 건수 | 용도 | 출처 |
|--------|------|------|------|
| 국내재해사례 API | 수천 건 | 사고 경위 텍스트 | https://www.data.go.kr/data/15121001/openapi.do |
| 산업재해통계 마이크로 데이터 | 수만 건 | 업종/발생형태/규모 분류 | https://www.data.go.kr/data/15127634/fileData.do |
| 안전보건법령 스마트검색 API | 법령 전체 | 관련 법령 매칭 | https://www.data.go.kr/data/15123696/openapi.do |
| KOSHA Guide 기술지침 | 800+ 종 | 안전수칙 정답 기준 | https://www.data.go.kr/data/15116595/fileData.do |

### A-2. 위반유형 분류 체계

SIF 데이터의 "재해유발요인" 컬럼 기반으로 아래 분류 체계 적용:

```
안전수칙 위반 유형
├── 보호구 관련
│   ├── 안전모 미착용
│   ├── 안전대 미착용
│   ├── 안전화 미착용
│   └── 보안경/방진마스크 미착용
├── 작업절차 관련
│   ├── 작업허가 미취득
│   ├── 잠금/태그아웃(LOTO) 미실시
│   ├── 안전난간/덮개 미설치
│   └── 신호수 미배치
├── 장비/시설 관련
│   ├── 안전장치 해제/무력화
│   ├── 장비 무자격 운전
│   └── 과적/과속
└── 환경/관리 관련
    ├── 가스 농도 미측정
    ├── 환기 미실시
    └── 안전교육 미실시
```

### A-3. 학습 데이터 포맷

```json
{
  "input": "6m 높이 철골 구조물에서 작업자가 안전대를 착용하지 않은 채 용접 작업 중 추락",
  "violation_type": "보호구_미착용",
  "sub_type": "안전대_미착용",
  "risk_level": "치명적",
  "cause": "안전대 미착용 + 안전난간 미설치",
  "law_reference": "산업안전보건법 제38조, 안전보건규칙 제44조",
  "countermeasure": "안전대 착용 의무화, 안전난간 설치, 추락방지망 설치",
  "industry": "건설업",
  "accident_type": "추락"
}
```

### A-4. 모델 학습 파이프라인

현재 프로젝트(`model.py`, `train_master.py`, `train_moe.py`) 구조를 그대로 활용.

#### Phase 1 — Master 모델 (Dense)

| 항목 | 값 |
|------|------|
| 학습 데이터 | SIF 재해개요 + 안전보건법령 텍스트 + KOSHA Guide |
| 목적 | 안전 도메인 언어 패턴 학습 |
| 아키텍처 | MasterBlock (Attention + 단일 FFN) |

#### Phase 2 — MoE 모델 (Sparse)

| 항목 | 값 |
|------|------|
| 학습 데이터 | SIF (위반유형 라벨 포함) + 재해사례 API |
| 목적 | 위반유형별 Expert 특화 |
| 아키텍처 | MoEBlock (Attention + Router + Experts) |
| Expert 역할 (기대) | Expert 0: 보호구 위반, Expert 1: 작업절차 위반, ... |

#### 3단계 MoE 학습

| Stage | 학습 대상 | 목적 |
|-------|----------|------|
| Stage 1 | Router만 | 위반유형별로 토큰을 적절한 Expert로 라우팅 |
| Stage 2 | Expert만 | 각 Expert가 특정 위반유형에 특화 |
| Stage 3 | Router + Expert | 전체 최적화 |

### A-5. 추론 (Inference)

```
입력: "작업자가 2m 높이 비계에서 안전대 미착용 상태로 작업 중 추락"

출력:
  - 위반유형: 보호구 미착용 (안전대)
  - 관련법령: 산업안전보건법 제38조 (안전조치)
             안전보건규칙 제44조 (안전대 부착설비 설치)
  - 위험등급: 치명적 (추락 사고)
  - 조치사항:
    1. 즉시 작업중지
    2. 안전대 지급 및 착용 확인
    3. 안전난간/추락방지망 설치 확인
    4. 관련 작업자 재교육 실시
```

---

## Phase B: 위반 사진 → 위반 식별 (추후)

### B-1. 사용 시나리오

```
안전팀이 현장 순회 중 위반 사항 촬영
  ↓
사진을 시스템에 업로드
  ↓
비전 모델이 사진 분석 (객체 탐지 + 분류)
  ├── "안전모 미착용 작업자 1명 감지"
  ├── "안전대 미착용 상태로 고소작업 중"
  └── "안전난간 미설치 구간 확인"
  ↓
감지 결과를 텍스트로 변환
  ↓
Phase A MoE 모델이 상세 분석
  ↓
최종 결과 출력:
  - 위반 항목 목록
  - 관련 법령 조항
  - 위험 등급
  - 시정 조치 사항
```

### B-2. 비전 모델 구성

| 단계 | 모델 | 역할 |
|------|------|------|
| 객체 탐지 | YOLO v8/v11 | 사진에서 작업자, 보호구, 장비, 시설물 검출 |
| 상태 분류 | CNN Classifier | 보호구 착용/미착용, 시설물 정상/미설치 판별 |
| 텍스트 변환 | Rule-based | 감지 결과를 구조화된 텍스트 설명으로 변환 |
| 상세 분석 | Phase A MoE | 위반유형 + 법령 + 조치 생성 |

### B-3. 비전 모델 학습 데이터

| 데이터셋 | 내용 | 출처 |
|----------|------|------|
| AI Hub 건설현장 위험 판단 데이터 | 5대 재해유형 이미지 + 라벨 | https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71407 |
| 서울시 안전모 객체 이미지 | 안전모 착용/미착용 이미지 | https://data.seoul.go.kr/dataList/OA-22789/L/1/datasetView.do |
| PPE Detection Dataset | 안전장비 객체탐지 (1,604장) | https://universe.roboflow.com/siabar/ppe-dataset-for-workplace-safety |
| Worker Safety Dataset | 작업자 안전장비 바운딩박스 | Roboflow (Worker-Safety) |

> 자체 데이터: 안전팀이 실제 현장에서 촬영한 위반 사진을 축적하면 도메인 특화 데이터로 추가 학습 가능.

### B-4. Phase A ↔ Phase B 연동 인터페이스

```python
class SafetyViolationAnalyzer:
    """위반 사진 → 위반 식별 통합 파이프라인"""
    def __init__(self, vision_model, text_model):
        self.vision_model = vision_model  # YOLO + Classifier
        self.text_model = text_model      # MoE 모델

    def analyze_image(self, image_path: str) -> dict:
        """위반 사진 한 장을 분석하여 위반 사항 식별"""
        # 1. 비전 모델로 사진 분석
        detections = self.vision_model.detect(image_path)
        # 예: [
        #   {"object": "worker", "helmet": False, "harness": False},
        #   {"object": "scaffold", "guardrail": False},
        # ]

        # 2. 감지 결과를 텍스트 설명으로 변환
        description = self._build_description(detections)
        # 예: "비계 위 작업자가 안전모와 안전대를 미착용한 상태.
        #      비계 안전난간 미설치 확인."

        # 3. Phase A 텍스트 모델로 상세 분석
        analysis = self.text_model.analyze(description)
        # 예: {
        #   "violations": ["안전모 미착용", "안전대 미착용", "안전난간 미설치"],
        #   "law_references": ["산안법 제38조", "안전보건규칙 제44조", ...],
        #   "risk_level": "치명적",
        #   "countermeasures": ["즉시 작업중지", "보호구 지급", ...],
        # }

        return {
            "image": image_path,
            "detections": detections,
            "description": description,
            "analysis": analysis,
        }
```

---

## 구현 순서

### 즉시 (Phase A)

1. [ ] 공공데이터포털에서 SIF 데이터 활용 신청 + 다운로드
2. [ ] SIF 데이터 전처리 스크립트 작성 (`prepare_safety_data.py`)
3. [ ] 위반유형 자동 라벨링 (재해유발요인 컬럼 기반)
4. [ ] Phase 1: Master 모델 학습 (안전 도메인)
5. [ ] Phase 2: MoE 모델 학습 (위반유형별 Expert 특화)
6. [ ] 추론 파이프라인 구현 + 테스트

### 추후 (Phase B)

7. [ ] 위반 사진 데이터 수집 (AI Hub + 자체 현장 촬영)
8. [ ] YOLO 모델 학습 (보호구/시설물 탐지)
9. [ ] 착용/미착용 분류 모델 학습
10. [ ] 비전 감지 → 텍스트 변환 모듈 구현
11. [ ] Phase A ↔ Phase B 연동 (SafetyViolationAnalyzer)
12. [ ] 안전팀용 웹/앱 인터페이스 (사진 업로드 → 결과 확인)

---

## 파일 구조 (예상)

```
progressive_moe/
├── model.py                      # 모델 아키텍처
├── utils.py                      # 공용 유틸리티
├── train_master.py               # Phase 1: Master 모델 학습
├── train_moe.py                  # Phase 2: MoE 학습
├── progressive_moe_layer.py      # 동적 Expert 확장
├── prepare_safety_data.py        # (신규) SIF 데이터 전처리
├── safety_inference.py           # (신규) 안전위반 추론 파이프라인
├── TRAINING_GUIDE.md             # 학습 가이드
├── SAFETY_AI_ROADMAP.md          # 본 문서
│
├── data/                         # (신규) 학습 데이터
│   ├── sif_raw/                  #   SIF 원본 CSV
│   ├── processed/                #   전처리된 학습 데이터
│   └── law_texts/                #   안전보건법령 텍스트
│
└── vision/                       # (추후) 이미지 위반 식별
    ├── train_detector.py         #   YOLO 보호구/시설물 탐지 학습
    ├── train_classifier.py       #   착용/미착용 분류 학습
    ├── analyzer.py               #   SafetyViolationAnalyzer 통합 파이프라인
    └── sample_images/            #   테스트용 위반 사진
```
