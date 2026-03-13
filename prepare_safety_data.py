"""
SIF(산업재해 고위험요인) 데이터 전처리 스크립트

입력: data/sif_raw/한국산업안전보건공단_산업재해 고위험요인(SIF) 목록_20241231.xlsx
출력: data/processed/sif_train.json       — 학습용 데이터
      data/processed/sif_stats.json       — 통계 요약
"""

import json
import re
import os
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl

# ==========================================
# 1. 위반유형 분류 체계 + 키워드 매핑
# ==========================================

# 대분류 → 소분류 → 매칭 키워드 (재해유발요인 텍스트에서 탐색)
VIOLATION_TAXONOMY = {
    "보호구_미착용": {
        "안전모_미착용": ["안전모", "보호모"],
        "안전대_미착용": ["안전대", "안전그네", "생명줄"],
        "안전화_미착용": ["안전화", "안전장화"],
        "보안경_방진마스크_미착용": ["보안경", "보호안경", "방진마스크", "방독마스크", "호흡보호구"],
    },
    "작업절차_위반": {
        "LOTO_미실시": [
            "정지시키지 않", "전원을 차단하지 않",
            "전원 차단", "잠금", "태그", "LOTO",
            "정비작업 중 작업자 외 근로자가 설비를 가동",
            "전원을 차단하지 않은",
        ],
        "안전난간_덮개_미설치": [
            "안전난간", "덮개 미설치", "개구부",
            "난간의 설치상태가 불량", "난간이 없는",
        ],
        "작업허가_미취득": ["작업허가", "허가 미취득", "작업계획서"],
        "신호수_미배치": ["신호수", "유도자", "유도원", "감시인"],
        "의사소통_미실시": ["의사소통", "신호체계", "통행을 통제하지 않"],
        "가스농도_미측정": [
            "가스농도", "산소농도", "산소·가소농도 미측정",
            "환기 미실시", "강제환기 미실시",
        ],
        "안전교육_미실시": ["안전교육", "특별교육"],
    },
    "장비시설_위반": {
        "안전장치_해제": [
            "안전장치가 설치되어 있지 않", "안전장치를 무력화",
            "방호장치", "리미트스위치", "안전장치 해제",
            "안전장치 미설치",
        ],
        "장비_무자격_운전": ["무자격", "무면허", "자격 미보유"],
        "장비_용도외_사용": ["주용도", "용도 외", "목적 외"],
        "과적_과속": ["과적", "과속", "과하중", "정격하중 초과"],
        "줄걸이_불량": [
            "줄걸이", "달기", "슬링", "와이어로프",
            "커넥트링", "샤클",
        ],
        "적재_불량": [
            "적재 상태가 불량", "적재상태가 불량",
            "무게가 한쪽으로 치우", "적재함 위에 올라",
        ],
        "설비_결함": [
            "균열·변형·부식", "균열", "부식", "마모·탈락",
            "용접 불량", "변형", "손상",
            "파손·파단", "파손되거나",
        ],
        "고정상태_불량": [
            "고정상태 불량", "안전블록·지지대 미설치",
            "지지대 미설치", "안전블록",
            "체결하지 않고 걸쳐놓은",
            "유·공압을 제거하지 않",
        ],
    },
    "추락_위험": {
        "불안전한_작업발판": [
            "밟고 있던 구조물이 파손", "균형을 잃기 쉬운 불안한 자세",
            "구조물을 밟고 올라", "설비를 밟고 올라",
            "뛰어넘어 이동",
        ],
        "사다리_불안전_사용": [
            "사다리 위에 올라가 작업", "사다리를 불안정하게",
            "아웃트리거", "사다리를 오르내려",
        ],
        "고소작업_미조치": [
            "높은 장소에서의 작업", "지붕", "설비 위",
        ],
        "굴착사면_붕괴": [
            "굴착사면", "사면이 무너", "매몰",
            "암석이 떨어", "부석", "전석",
            "막장면",
        ],
    },
    "위험물질_관련": {
        "점화원_관리_미흡": [
            "점화원", "아크", "정전기", "용접불티",
            "인화성", "가연성", "충전부",
        ],
        "유해물질_누출": [
            "누출", "배관·호스", "용기", "가스누출",
        ],
        "밀폐공간_미조치": [
            "밀폐", "맨홀", "피트", "오·폐수",
            "질식", "유해가스",
        ],
    },
    "차량_운행_위반": {
        "이동동선_미분리": [
            "이동동선", "통행로를 구분하지 않",
            "혼재작업",
        ],
        "사각지대_미확인": [
            "사각지대", "시야가 확보되지 않",
            "진행방향", "후진",
            "운전석 이탈", "차량이 움직이는 것을 방지",
        ],
        "운행경로_위험": [
            "운행경로", "급경사", "커브", "지반",
            "전도", "미끄러",
        ],
        "차량_정비_미흡": [
            "차량의 상태를 유지·관리", "점검하지 않아",
        ],
    },
    "통행_전도_위험": {
        "바닥_미끄럼": [
            "미끄러운 상태의 바닥", "오일", "빙판",
            "바닥의 요철",
        ],
        "통행로_장애물": [
            "어두운 장소", "요철부에 걸려",
            "통행", "이동 중",
            "위험요소", "낭떠러지",
        ],
    },
    "벌목_작업_위험": {
        "벌도목_위험": [
            "벌도목", "벌목", "조재", "수목에 걸려",
            "의도하지 않은 방향으로 쓰러",
        ],
    },
    "가공설비_위험": {
        "가공물_비래": [
            "튕겨나온 가공날", "튕겨나온 가공물",
            "고속회전하면서", "휘어진 가공물",
        ],
    },
}


def classify_violation(factor_text: str, overview_text: str = "") -> dict:
    """재해유발요인 + 재해개요 텍스트에서 위반유형 분류.

    Returns:
        {
            "violation_type": 대분류,
            "sub_type": 소분류,
            "confidence": "keyword",  # 매칭 방법
        }
    """
    combined = (factor_text or "") + " " + (overview_text or "")
    combined = combined.strip()

    best_match = None
    best_keyword_len = 0  # 더 긴 키워드 매칭 우선

    for major, sub_dict in VIOLATION_TAXONOMY.items():
        for sub, keywords in sub_dict.items():
            for kw in keywords:
                if kw in combined:
                    if len(kw) > best_keyword_len:
                        best_keyword_len = len(kw)
                        best_match = {
                            "violation_type": major,
                            "sub_type": sub,
                            "confidence": "keyword",
                        }

    if best_match:
        return best_match

    # 매칭 실패 시 재해개요에서 추가 힌트 탐색
    fallback_map = [
        ("추락", "추락_위험", "고소작업_미조치"),
        ("끼임", "장비시설_위반", "안전장치_해제"),
        ("넘어", "통행_전도_위험", "바닥_미끄럼"),
        ("충돌", "차량_운행_위반", "사각지대_미확인"),
        ("감전", "위험물질_관련", "점화원_관리_미흡"),
        ("질식", "위험물질_관련", "밀폐공간_미조치"),
        ("화재", "위험물질_관련", "점화원_관리_미흡"),
        ("폭발", "위험물질_관련", "점화원_관리_미흡"),
        ("낙하", "장비시설_위반", "적재_불량"),
        ("붕괴", "장비시설_위반", "설비_결함"),
        ("전도", "차량_운행_위반", "운행경로_위험"),
        ("절단", "장비시설_위반", "안전장치_해제"),
    ]
    for hint, major, sub in fallback_map:
        if hint in combined:
            return {
                "violation_type": major,
                "sub_type": sub,
                "confidence": "fallback",
            }

    return {
        "violation_type": "미분류",
        "sub_type": "미분류",
        "confidence": "none",
    }


# ==========================================
# 2. 위험등급 추정
# ==========================================

def estimate_risk_level(overview: str) -> str:
    """재해개요에서 위험등급 추정."""
    text = overview or ""
    if any(kw in text for kw in ["사망", "익사", "질식사"]):
        return "치명적"
    if any(kw in text for kw in ["중상", "골절", "절단", "화상", "의식불명"]):
        return "높음"
    if any(kw in text for kw in ["부상", "찰과상", "타박상"]):
        return "보통"
    # SIF 데이터 자체가 사망사고 분석이므로 기본값 '치명적'
    return "치명적"


# ==========================================
# 3. 사고유형 추정
# ==========================================

def estimate_accident_type(overview: str, factor: str = "") -> str:
    """재해개요/재해유발요인에서 사고유형 추정."""
    combined = (overview or "") + " " + (factor or "")
    type_map = [
        ("추락", "추락"),
        ("넘어", "전도"),
        ("전도", "전도"),
        ("끼임", "끼임"),
        ("끼어", "끼임"),
        ("감전", "감전"),
        ("질식", "질식"),
        ("익사", "익사"),
        ("화재", "화재"),
        ("폭발", "폭발"),
        ("충돌", "충돌"),
        ("부딪", "충돌"),
        ("낙하", "낙하·비래"),
        ("날아", "낙하·비래"),
        ("맞아", "낙하·비래"),
        ("무너", "붕괴"),
        ("붕괴", "붕괴"),
        ("절단", "절단·베임"),
        ("베임", "절단·베임"),
        ("화상", "화상"),
        ("깔림", "깔림"),
        ("깔려", "깔림"),
    ]
    for kw, atype in type_map:
        if kw in combined:
            return atype
    return "기타"


# ==========================================
# 4. 엑셀 파싱
# ==========================================

def parse_manufacturing_sheet(ws) -> list[dict]:
    """제조업 등(건설업 외 업종) 시트 파싱.

    헤더(행12): 고유번호, 산재업종(대), 산재업종(중), 산재업종(소),
                재해개요, 기인물, 고위험작업·상황, 재해유발요인, 위험성감소대책
    데이터: 행13~
    """
    records = []
    for row in ws.iter_rows(min_row=14, values_only=True):
        # 빈 행 건너뛰기
        if not row[0] or not row[4]:
            continue

        uid = str(row[0]).strip()
        industry_major = str(row[1] or "").strip()
        industry_mid = str(row[2] or "").strip()
        industry_minor = str(row[3] or "").strip()
        overview = str(row[4] or "").strip()
        equipment = str(row[5] or "").strip()
        high_risk_task = str(row[6] or "").strip()
        cause_factor = str(row[7] or "").strip()
        countermeasure = str(row[8] or "").strip()

        # 『...』 안의 핵심 위반사항 추출
        bracket_factors = re.findall(r"『(.+?)』", cause_factor)

        # 위반유형 분류
        classification = classify_violation(cause_factor, overview)

        records.append({
            "id": uid,
            "source": "제조업",
            "industry": industry_major,
            "industry_detail": f"{industry_mid} > {industry_minor}" if industry_mid else "",
            "high_risk_task": high_risk_task,
            "overview": overview,
            "equipment": equipment,
            "cause_factor": cause_factor,
            "cause_factor_key": bracket_factors[0] if bracket_factors else "",
            "countermeasure": countermeasure,
            "violation_type": classification["violation_type"],
            "sub_type": classification["sub_type"],
            "confidence": classification["confidence"],
            "risk_level": estimate_risk_level(overview),
            "accident_type": estimate_accident_type(overview, cause_factor),
        })

    return records


def parse_construction_sheet(ws) -> list[dict]:
    """건설업 시트 파싱.

    헤더(행2-3): 고유번호, 공종, 작업명, 단위작업명,
                 재해개요, 기인물, 재해유발요인, 위험성감소대책
    데이터: 행4~
    """
    records = []
    for row in ws.iter_rows(min_row=5, values_only=True):
        if not row[0] or not row[4]:
            continue

        uid = str(row[0]).strip()
        work_type = str(row[1] or "").strip()       # 공종
        work_name = str(row[2] or "").strip()       # 작업명
        work_detail = str(row[3] or "").strip()     # 단위작업명
        overview = str(row[4] or "").strip()
        equipment = str(row[5] or "").strip()
        cause_factor = str(row[6] or "").strip()
        countermeasure = str(row[7] or "").strip()

        # 건설업은 『...』가 없는 자유 텍스트
        bracket_factors = re.findall(r"『(.+?)』", cause_factor)

        classification = classify_violation(cause_factor, overview)

        high_risk_task = " > ".join(
            p for p in [work_type, work_name, work_detail] if p
        )

        records.append({
            "id": uid,
            "source": "건설업",
            "industry": "건설업",
            "industry_detail": work_type,
            "high_risk_task": high_risk_task,
            "overview": overview,
            "equipment": equipment,
            "cause_factor": cause_factor,
            "cause_factor_key": bracket_factors[0] if bracket_factors else "",
            "countermeasure": countermeasure,
            "violation_type": classification["violation_type"],
            "sub_type": classification["sub_type"],
            "confidence": classification["confidence"],
            "risk_level": estimate_risk_level(overview),
            "accident_type": estimate_accident_type(overview, cause_factor),
        })

    return records


# ==========================================
# 5. 학습 데이터 포맷 변환
# ==========================================

def to_training_format(record: dict) -> dict:
    """SAFETY_AI_ROADMAP.md의 학습 데이터 포맷으로 변환."""
    return {
        "id": record["id"],
        "input": record["overview"],
        "violation_type": record["violation_type"],
        "sub_type": record["sub_type"],
        "risk_level": record["risk_level"],
        "cause": record["cause_factor_key"] or record["cause_factor"],
        "countermeasure": record["countermeasure"],
        "industry": record["industry"],
        "industry_detail": record["industry_detail"],
        "accident_type": record["accident_type"],
        "equipment": record["equipment"],
        "high_risk_task": record["high_risk_task"],
        "source": record["source"],
        "confidence": record["confidence"],
    }


# ==========================================
# 6. 통계 생성
# ==========================================

def generate_stats(records: list[dict]) -> dict:
    """데이터 통계 요약 생성."""
    stats = {
        "total": len(records),
        "by_source": dict(Counter(r["source"] for r in records)),
        "by_violation_type": dict(Counter(r["violation_type"] for r in records).most_common()),
        "by_sub_type": dict(Counter(r["sub_type"] for r in records).most_common()),
        "by_accident_type": dict(Counter(r["accident_type"] for r in records).most_common()),
        "by_risk_level": dict(Counter(r["risk_level"] for r in records)),
        "by_confidence": dict(Counter(r["confidence"] for r in records)),
        "by_industry": dict(Counter(r["industry"] for r in records).most_common(20)),
        "by_equipment_top20": dict(Counter(r["equipment"] for r in records).most_common(20)),
    }
    return stats


# ==========================================
# 7. 메인 실행
# ==========================================

def main():
    base_dir = Path(__file__).parent
    xlsx_path = base_dir / "data" / "sif_raw" / "한국산업안전보건공단_산업재해 고위험요인(SIF) 목록_20241231.xlsx"
    out_dir = base_dir / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f">>> SIF 데이터 전처리 시작")
    print(f"   입력: {xlsx_path}")

    # 엑셀 로드
    wb = openpyxl.load_workbook(str(xlsx_path), read_only=True)

    # 제조업 파싱
    print("\n>>> 제조업 시트 파싱 중...")
    mfg_records = parse_manufacturing_sheet(wb["제조업 등(건설업 외 업종)"])
    print(f"   -> {len(mfg_records)}건 파싱 완료")

    # 건설업 파싱
    print("\n>>> 건설업 시트 파싱 중...")
    con_records = parse_construction_sheet(wb["건설업"])
    print(f"   -> {len(con_records)}건 파싱 완료")

    wb.close()

    # 전체 합치기
    all_records = mfg_records + con_records
    print(f"\n>>> 전체: {len(all_records)}건")

    # 학습 데이터 포맷 변환
    train_data = [to_training_format(r) for r in all_records]

    # 통계 생성
    stats = generate_stats(all_records)

    # 저장
    train_path = out_dir / "sif_train.json"
    stats_path = out_dir / "sif_stats.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"\n>>> 학습 데이터 저장: {train_path}")

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f">>> 통계 데이터 저장: {stats_path}")

    # 결과 요약 출력
    print(f"\n{'='*60}")
    print(f"전처리 완료 요약")
    print(f"{'='*60}")
    print(f"총 레코드: {stats['total']}건")
    print(f"  제조업: {stats['by_source'].get('제조업', 0)}건")
    print(f"  건설업: {stats['by_source'].get('건설업', 0)}건")

    print(f"\n위반유형 분류:")
    for vtype, count in stats["by_violation_type"].items():
        print(f"  {vtype}: {count}건 ({count/stats['total']*100:.1f}%)")

    print(f"\n분류 신뢰도:")
    for conf, count in stats["by_confidence"].items():
        print(f"  {conf}: {count}건 ({count/stats['total']*100:.1f}%)")

    print(f"\n사고유형:")
    for atype, count in list(stats["by_accident_type"].items())[:10]:
        print(f"  {atype}: {count}건")


if __name__ == "__main__":
    main()
