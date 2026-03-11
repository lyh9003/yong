from bs4 import BeautifulSoup
import requests
import datetime
from tqdm import tqdm
import time
import pandas as pd
import random
from openai import OpenAI
import os
from rapidfuzz import fuzz

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

MOBILE_NEWS_PREFIX = "https://n.news.naver.com/mnews/article/"

CONTAINER_SELECTOR = (
    "div.sds-comps-horizontal-layout"
    ".sds-comps-full-layout"
    ".sds-comps-profile"
    ".type-basic"
    ".size-lg"
    ".title-color-g10 "
    f"a[href^='{MOBILE_NEWS_PREFIX}']"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.naver.com/",
}


# ====== 파일 I/O ======

def load_existing_links(file_name: str) -> set:
    """기존 CSV에서 이미 수집된 링크 집합을 반환."""
    if os.path.exists(file_name):
        df = pd.read_csv(file_name, encoding='utf-8-sig', usecols=['link'])
        return set(df['link'].dropna().tolist())
    return set()


def load_existing_data(file_name: str) -> pd.DataFrame:
    if os.path.exists(file_name):
        return pd.read_csv(file_name, encoding='utf-8-sig')
    return pd.DataFrame()


def merge_and_remove_duplicates(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset='link', keep='first').reset_index(drop=True)
        print(f"병합 완료: 최종 {len(combined)}개")
        return combined
    print(f"새 데이터만 사용: {len(new_df)}개")
    return new_df


def save_updated_data(data: pd.DataFrame, file_name: str) -> None:
    data.to_csv(file_name, encoding='utf-8-sig', index=False)
    print(f"저장 완료: {file_name}")


# ====== 크롤링 ======

def makePgNum(num: int) -> int:
    if num == 1:
        return num
    elif num == 0:
        return num + 1
    return num + 9 * (num - 1)


def makeUrl(search: str, start_pg: int, end_pg: int, start_date: str, end_date: str) -> list[str]:
    return [
        (
            f"https://search.naver.com/search.naver?where=news&sm=tab_opt"
            f"&sort=0&photo=0&field=0&pd=3&ds={start_date}&de={end_date}"
            f"&query={search}&start={makePgNum(i)}"
        )
        for i in range(start_pg, end_pg + 1)
    ]


def articles_crawler(search_page_url: str) -> list[str]:
    """네이버 뉴스 검색 결과 페이지에서 기사 링크 목록 반환."""
    html = requests.get(search_page_url, headers=HEADERS, timeout=10).text
    soup = BeautifulSoup(html, "html.parser")
    links = {
        a["href"].split("?", 1)[0]
        for a in soup.select(CONTAINER_SELECTOR)
    }
    return sorted(links)


def crawl_article(url: str) -> dict:
    """단일 기사 URL에서 제목, 본문, 날짜, 언론사를 추출."""
    soup = BeautifulSoup(requests.get(url, headers=HEADERS, timeout=10).text, "html.parser")

    img = soup.select_one(
        "#ct > div.media_end_head.go_trans > div.media_end_head_top > a.media_end_head_top_logo > img"
    )
    company = img.attrs['title'] if img else "정보 없음"

    title_el = soup.select_one(
        "#ct > div.media_end_head.go_trans > div.media_end_head_title > h2"
    )
    title = title_el.text.strip() if title_el else "제목 없음"

    body = soup.find("div", class_="newsct_article _article_body")
    content = body.get_text(strip=True) if body else ""

    date_el = soup.select_one("span.media_end_head_info_datestamp_time")
    news_date = date_el.attrs.get('data-date-time', "날짜 없음") if date_el else "날짜 없음"

    return {"company": company, "title": title, "content": content, "date": news_date}


# ====== GPT 처리 ======

def is_related_to_semiconductor(title: str, content_snippet: str) -> bool:
    """제목 + 본문 앞부분을 보고 반도체 산업 관련 기사인지 판단."""
    prompt = (
        "다음 기사가 반도체 산업(메모리, 파운드리, AI 반도체, 반도체 장비·소재, "
        "공급망, 지정학적 규제, 관련 기업 실적 등)과 관련이 있으면 'YES', 없으면 'NO'로만 답하세요.\n\n"
        f"제목: {title}\n"
        f"본문 앞부분: {content_snippet[:400]}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "반도체 산업 관련성 판단기. YES 또는 NO만 출력."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=5,
        )
        return response.choices[0].message.content.strip().upper() == "YES"
    except Exception as e:
        print(f"관련성 판단 오류: {e}")
        return False


def get_related_keyword(title: str, keywords: list[str]) -> str | None:
    """기사 제목에 가장 잘 맞는 키워드 하나를 반환. 없으면 None."""
    prompt = (
        "다음 기사 제목과 가장 관련 있는 키워드를 아래 목록에서 딱 하나만 골라 그대로 출력하세요.\n"
        "목록에 없으면 '관련 없음'이라고만 답하세요.\n\n"
        f"제목: {title}\n\n"
        f"키워드 목록:\n" + "\n".join(keywords)
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "키워드 분류기. 키워드 하나 또는 '관련 없음'만 출력."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=20,
        )
        answer = response.choices[0].message.content.strip()
        return None if answer == "관련 없음" else answer
    except Exception as e:
        print(f"키워드 분류 오류: {e}")
        return None


def summarize_content(text: str) -> str:
    """기사 본문을 200자 이내로 요약."""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "뉴스 기사 요약기."},
                {"role": "user", "content": f"다음 기사를 200자 이내로 요약해주세요:\n\n{text[:3000]}"}
            ],
            temperature=0.5,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"요약 오류: {e}")
        return "요약 실패"


# ====== 중복 제거 ======

def deduplicate_by_title_similarity(df: pd.DataFrame, threshold: int = 85) -> pd.DataFrame:
    """rapidfuzz로 제목 유사도 기반 중복 제거. 본문이 긴 쪽을 남긴다."""
    titles = df["title"].tolist()
    to_remove = set()

    for i in range(len(titles)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(titles)):
            if j in to_remove:
                continue
            if fuzz.token_sort_ratio(titles[i], titles[j]) >= threshold:
                len_i = len(df.iloc[i]["content"])
                len_j = len(df.iloc[j]["content"])
                to_remove.add(j if len_i >= len_j else i)

    kept = df.drop(index=list(to_remove)).reset_index(drop=True)
    print(f"유사 제목 중복 제거: {len(to_remove)}개 제거 → {len(kept)}개 남음")
    return kept


# ====== 메인 ======

def main():
    file_name = "Total_Filtered_No_Comment.csv"
    existing_links = load_existing_links(file_name)
    print(f"기존 수집 링크 수: {len(existing_links)}")

    keyword_df = pd.read_csv('keyword_org.csv', encoding='utf-8-sig')
    keywords = keyword_df['키워드'].unique().tolist()

    end_date = datetime.datetime.now().strftime("%Y.%m.%d")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime("%Y.%m.%d")

    start_pg, end_pg = 1, 2
    all_news_df = pd.DataFrame()

    for search in keywords:
        print(f"\n=== 키워드 '{search}' ===")
        urls = makeUrl(search, start_pg, end_pg, start_date, end_date)

        # 1. 링크 수집
        raw_links = []
        for url in urls:
            raw_links.extend(articles_crawler(url))
            time.sleep(random.uniform(0.2, 0.5))

        # 2. 이미 수집된 링크 제외 (GPT 호출 전 필터링 → API 비용 절감)
        new_links = list({link for link in raw_links if link not in existing_links})
        print(f"신규 링크: {len(new_links)}개 (전체 {len(raw_links)}개 중)")
        if not new_links:
            continue

        # 3. 기사 본문 크롤링
        articles = []
        for link in tqdm(new_links, desc="기사 크롤링"):
            try:
                data = crawl_article(link)
                data["link"] = link
                articles.append(data)
            except Exception as e:
                print(f"크롤링 실패 {link}: {e}")
            time.sleep(random.uniform(0.2, 0.4))

        if not articles:
            continue

        news_df = pd.DataFrame(articles)

        # 4. 반도체 관련성 필터 (제목 + 본문 앞부분으로 판단)
        news_df['관련성'] = news_df.apply(
            lambda r: is_related_to_semiconductor(r['title'], r['content']), axis=1
        )
        news_df = news_df[news_df['관련성']].reset_index(drop=True)
        print(f"관련성 필터 후: {len(news_df)}개")
        if news_df.empty:
            continue

        # 5. 키워드 분류
        news_df['키워드'] = news_df['title'].apply(lambda t: get_related_keyword(t, keywords))
        news_df = news_df[news_df['키워드'].notna()].reset_index(drop=True)
        print(f"키워드 분류 후: {len(news_df)}개")
        if news_df.empty:
            continue

        # 6. 제목 유사도 중복 제거
        news_df = deduplicate_by_title_similarity(news_df)

        # 7. 요약 생성
        news_df['summary'] = news_df['content'].apply(
            lambda x: summarize_content(x) if len(x) > 10 else "내용 부족"
        )

        news_df['검색어'] = search
        all_news_df = pd.concat([all_news_df, news_df], ignore_index=True)
        print(f"키워드 '{search}' 완료: {len(news_df)}개 추가")

    if all_news_df.empty:
        print("새로 수집된 기사가 없습니다.")
        return

    # 전체 링크 중복 제거 후 제목 유사도 중복 제거
    all_news_df = all_news_df.drop_duplicates(subset='link', keep='first', ignore_index=True)
    all_news_df = deduplicate_by_title_similarity(all_news_df)

    all_news_df['date'] = pd.to_datetime(all_news_df['date'], errors='coerce').dt.strftime('%Y-%m-%d')

    existing_data = load_existing_data(file_name)
    updated_data = merge_and_remove_duplicates(existing_data, all_news_df)
    save_updated_data(updated_data, file_name)


if __name__ == "__main__":
    main()
