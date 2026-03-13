"""
안전보건법령 텍스트 수집 스크립트

국가법령정보센터 Open API(DRF)를 사용하여 안전보건 관련 법령 조문을 다운로드하고
학습용 텍스트로 변환합니다.

출력:
  data/law_texts/{법령명}.json       — 법령별 조문 전체
  data/law_texts/all_laws.json       — 전체 법령 통합 (학습용)
  data/law_texts/law_stats.json      — 통계 요약
"""

import json
import re
import os
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import quote


# ==========================================
# 1. 수집 대상 법령 목록
# ==========================================

# (MST 법령일련번호, 법령명, 약칭)
TARGET_LAWS = [
    (276853, "산업안전보건법", "산안법"),
    (277411, "산업안전보건법 시행령", "산안법 시행령"),
    (271485, "산업안전보건법 시행규칙", "산안법 시행규칙"),
    (273603, "산업안전보건기준에 관한 규칙", "안전보건규칙"),
    (228817, "중대재해 처벌 등에 관한 법률", "중대재해처벌법"),
    (277417, "중대재해 처벌 등에 관한 법률 시행령", "중대재해처벌법 시행령"),
]

DRF_BASE = "https://www.law.go.kr/DRF/lawService.do"
OC = "chetera"  # 공개 테스트 계정


# ==========================================
# 2. XML 파싱
# ==========================================

def fetch_law_xml(mst: int) -> str:
    """법령 XML 다운로드."""
    url = f"{DRF_BASE}?OC={OC}&target=law&MST={mst}&type=XML"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def parse_articles(xml_text: str) -> list[dict]:
    """XML에서 조문 목록 추출."""
    articles = []

    # 조문단위 블록 추출
    jo_units = re.findall(
        r"<조문단위[^>]*>(.*?)</조문단위>", xml_text, re.DOTALL
    )

    for jo in jo_units:
        num_match = re.findall(r"<조문번호>(\d+)</조문번호>", jo)
        title_match = re.findall(
            r"<조문제목><!\[CDATA\[(.+?)\]\]></조문제목>", jo
        )
        # 조문내용은 여러 CDATA 블록일 수 있음
        content_parts = re.findall(
            r"<조문내용>(.*?)</조문내용>", jo, re.DOTALL
        )
        content_text = ""
        if content_parts:
            # CDATA 내부 텍스트 추출
            cdatas = re.findall(r"<!\[CDATA\[(.+?)\]\]>", content_parts[0], re.DOTALL)
            content_text = "\n".join(c.strip() for c in cdatas if c.strip())

        # 항 추출
        hang_units = re.findall(
            r"<항단위[^>]*>(.*?)</항단위>", jo, re.DOTALL
        )
        hang_texts = []
        for hang in hang_units:
            hang_content = re.findall(
                r"<항내용><!\[CDATA\[(.+?)\]\]>", hang, re.DOTALL
            )
            if hang_content:
                hang_texts.append(hang_content[0].strip())

            # 호 추출
            ho_units = re.findall(r"<호단위[^>]*>(.*?)</호단위>", hang, re.DOTALL)
            for ho in ho_units:
                ho_content = re.findall(
                    r"<호내용><!\[CDATA\[(.+?)\]\]>", ho, re.DOTALL
                )
                if ho_content:
                    hang_texts.append("  " + ho_content[0].strip())

                # 목 추출
                mok_units = re.findall(r"<목단위[^>]*>(.*?)</목단위>", ho, re.DOTALL)
                for mok in mok_units:
                    mok_content = re.findall(
                        r"<목내용><!\[CDATA\[(.+?)\]\]>", mok, re.DOTALL
                    )
                    if mok_content:
                        hang_texts.append("    " + mok_content[0].strip())

        # 전체 텍스트 조합
        full_text = content_text
        if hang_texts:
            full_text = (full_text + "\n" if full_text else "") + "\n".join(hang_texts)

        if not full_text.strip():
            continue

        article_num = int(num_match[0]) if num_match else 0
        article_title = title_match[0].strip() if title_match else ""

        articles.append({
            "article_num": article_num,
            "article_title": article_title,
            "text": full_text.strip(),
        })

    return articles


def parse_law_metadata(xml_text: str) -> dict:
    """XML에서 법령 메타데이터 추출."""
    def extract(tag):
        match = re.findall(
            rf"<{tag}><!\[CDATA\[(.+?)\]\]></{tag}>", xml_text
        )
        if not match:
            match = re.findall(rf"<{tag}>(.+?)</{tag}>", xml_text)
        return match[0].strip() if match else ""

    return {
        "law_name": extract("법령명_한글"),
        "law_id": extract("법령ID"),
        "promulgation_date": extract("공포일자"),
        "enforcement_date": extract("시행일자"),
    }


# ==========================================
# 3. 학습용 텍스트 변환
# ==========================================

def article_to_training_text(law_name: str, short_name: str, article: dict) -> str:
    """조문 하나를 학습용 플레인 텍스트로 변환."""
    num = article["article_num"]
    title = article["article_title"]
    text = article["text"]

    header = f"[{short_name}] 제{num}조"
    if title:
        header += f" ({title})"

    return f"{header}\n{text}"


# ==========================================
# 4. 메인 실행
# ==========================================

def main():
    base_dir = Path(__file__).parent
    out_dir = base_dir / "data" / "law_texts"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(">>> 안전보건법령 텍스트 수집 시작")
    print(f"   출력 디렉토리: {out_dir}")

    all_articles = []  # 전체 통합
    stats = {
        "laws": [],
        "total_articles": 0,
        "total_chars": 0,
    }

    for mst, law_name, short_name in TARGET_LAWS:
        print(f"\n>>> [{short_name}] 다운로드 중... (MST={mst})")

        try:
            xml = fetch_law_xml(mst)
        except Exception as e:
            print(f"   -> 다운로드 실패: {e}")
            continue

        # 메타데이터
        metadata = parse_law_metadata(xml)
        print(f"   -> 법령명: {metadata['law_name']}")
        print(f"   -> 시행일자: {metadata['enforcement_date']}")

        # 조문 파싱
        articles = parse_articles(xml)
        print(f"   -> 조문 수: {len(articles)}개")

        if not articles:
            print("   -> 조문 없음, 건너뜀")
            continue

        # 법령별 JSON 저장
        law_data = {
            "law_name": law_name,
            "short_name": short_name,
            "metadata": metadata,
            "articles": articles,
        }

        safe_name = law_name.replace(" ", "_")
        law_path = out_dir / f"{safe_name}.json"
        with open(law_path, "w", encoding="utf-8") as f:
            json.dump(law_data, f, ensure_ascii=False, indent=2)
        print(f"   -> 저장: {law_path.name}")

        # 학습용 텍스트 변환
        for art in articles:
            training_text = article_to_training_text(law_name, short_name, art)
            all_articles.append({
                "law_name": law_name,
                "short_name": short_name,
                "article_num": art["article_num"],
                "article_title": art["article_title"],
                "text": training_text,
                "raw_text": art["text"],
            })

        # 통계
        total_chars = sum(len(a["text"]) for a in articles)
        stats["laws"].append({
            "law_name": law_name,
            "short_name": short_name,
            "num_articles": len(articles),
            "total_chars": total_chars,
            "enforcement_date": metadata["enforcement_date"],
        })
        stats["total_articles"] += len(articles)
        stats["total_chars"] += total_chars

        # API 부하 방지
        time.sleep(1)

    # 전체 통합 저장
    all_path = out_dir / "all_laws.json"
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)
    print(f"\n>>> 전체 통합 저장: {all_path} ({len(all_articles)}개 조문)")

    # 통계 저장
    stats_path = out_dir / "law_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f">>> 통계 저장: {stats_path}")

    # 결과 요약
    print(f"\n{'='*60}")
    print(f"법령 텍스트 수집 완료")
    print(f"{'='*60}")
    print(f"총 법령: {len(stats['laws'])}개")
    print(f"총 조문: {stats['total_articles']}개")
    print(f"총 문자: {stats['total_chars']:,}자")
    print()
    for law in stats["laws"]:
        print(f"  {law['short_name']:20s} | {law['num_articles']:4d}개 조문 | {law['total_chars']:>8,}자")


if __name__ == "__main__":
    main()
