from bs4 import BeautifulSoup
import requests
import datetime
from tqdm import tqdm
import time
import pandas as pd
import random
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from openai import OpenAI
import re
import json
import os

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# ====== 뉴스 크롤링 ======

# 기존 데이터 불러오기
def load_existing_data(file_name):
    if os.path.exists(file_name):
        existing_df = pd.read_csv(file_name, encoding='utf-8-sig')
        print(f"기존 파일 로드 완료: {file_name}")
    else:
        existing_df = pd.DataFrame()
        print("기존 파일이 없습니다. 새로운 데이터를 생성합니다.")
    return existing_df

# 기존 데이터와 새 데이터 병합
def merge_and_remove_duplicates(existing_df, new_df):
    if not existing_df.empty:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset='link', keep='first').reset_index(drop=True)
        print(f"기존 데이터와 새 데이터를 병합했습니다. 최종 데이터 개수: {len(combined_df)}")
    else:
        combined_df = new_df
        print(f"새 데이터만 사용합니다. 데이터 개수: {len(combined_df)}")
    return combined_df

# 파일 저장
def save_updated_data(data, file_name):
    data.to_csv(file_name, encoding='utf-8-sig', index=False)
    print(f"업데이트된 데이터를 저장했습니다: {file_name}")

# 페이지 URL 변환
def makePgNum(num):
    if num == 1:
        return num
    elif num == 0:
        return num + 1
    else:
        return num + 9 * (num - 1)

# 크롤링할 URL 생성
def makeUrl(search, start_pg, end_pg, start_date, end_date):
    urls = []
    for i in range(start_pg, end_pg + 1):
        page = makePgNum(i)
        url = f"https://search.naver.com/search.naver?where=news&sm=tab_opt&sort=0&photo=0&field=0&pd=3&ds={start_date}&de={end_date}&query={search}&start={page}"
        urls.append(url)
    return urls

# HTML 속성 추출
def news_attrs_crawler(articles, attrs):
    attrs_content = []
    for i in articles:
        attrs_content.append(i.attrs[attrs])
    return attrs_content

# 뉴스 링크 크롤링
#def articles_crawler(url):
#    original_html = requests.get(url, headers=headers)
#    html = BeautifulSoup(original_html.text, "html.parser")#
#
#    url_naver = html.select(
#        "div.group_news > ul.list_news > li div.news_area > div.news_info > div.info_group > a.info")
#    url = news_attrs_crawler(url_naver, 'href')
#    return url

def articles_crawler(url):
    original_html = requests.get(url, headers=headers)
    
    # 응답 HTML 확인: GitHub Actions 로그에 출력
    print("==== HTML Response 시작 ====")
    print(original_html.text)
    print("==== HTML Response 끝 ====")
    
    html = BeautifulSoup(original_html.text, "html.parser")
    url_naver = html.select(
        "div.group_news > ul.list_news > li div.news_area > div.news_info > div.info_group > a.info")
    url = news_attrs_crawler(url_naver, 'href')
    return url

def makeList(newlist, content):
    for i in content:
        for j in i:
            newlist.append(j)
    return newlist



# 키워드 파일 불러오기
keyword_df = pd.read_csv('keyword_org.csv', encoding='cp949')
keywords = keyword_df['키워드'].unique().tolist()


# 날짜 계산 (오늘 기준 최신 1주일)
end_date = datetime.datetime.now().strftime("%Y.%m.%d")
start_date = (datetime.datetime.now() - datetime.timedelta(days=31)).strftime("%Y.%m.%d")

# 사용자 입력 -----> 자동으로 불러오도록 변경!
#search = input("검색할 키워드를 입력해주세요: ")
#start_pg = int(input("크롤링할 시작 페이지를 입력해주세요 (숫자만): "))
#end_pg = int(input("크롤링할 종료 페이지를 입력해주세요 (숫자만): "))

# 시작 및 종료 페이지 설정
start_pg = 1
end_pg = 2  # 원하는 페이지 범위로 설정하세요.

# URL 생성
#headers = {
#    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
#}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.naver.com/"
}

# 결과를 저장할 DataFrame 초기화
all_news_df = pd.DataFrame()

# 키워드별로 반복
for search in keywords:
    print(f"\n=== 키워드 '{search}'에 대한 뉴스 수집 시작 ===")
    
    urls = makeUrl(search, start_pg, end_pg, start_date, end_date)
    
    # 뉴스 데이터 수집
    news_companies = []
    news_titles = []
    news_url = []
    news_contents = []
    news_dates = []
    
    for i in urls:
        url_list = articles_crawler(i)
        news_url.append(url_list)
        time.sleep(random.uniform(0.2, 0.4))
    
    news_url_1 = []
    makeList(news_url_1, news_url)
    
    # NAVER 뉴스만 필터링
    final_urls = []
    for i in range(len(news_url_1)):
        if "news.naver.com" in news_url_1[i]:
            final_urls.append(news_url_1[i])
    
    # 뉴스 내용 크롤링
    for i in tqdm(final_urls):
        news = requests.get(i, headers=headers)
        news_html = BeautifulSoup(news.text, "html.parser")
    
        # 언론사 이름
        html_company = news_html.select_one(
            "#ct > div.media_end_head.go_trans > div.media_end_head_top > a.media_end_head_top_logo > img")
        company = html_company.attrs['title'] if html_company else "정보 없음"
    
        # 뉴스 제목
        title = news_html.select_one("#ct > div.media_end_head.go_trans > div.media_end_head_title > h2")
        title = title.text.strip() if title else "제목 없음"
    
        # 뉴스 본문
        content = news_html.find("div", class_="newsct_article _article_body")
        content = content.get_text(strip=True) if content else "내용 없음"
    
        # 뉴스 날짜
        try:
            html_date = news_html.select_one(
                "div#ct > div.media_end_head.go_trans > div.media_end_head_info.nv_notrans > div.media_end_head_info_datestamp > div > span")
            news_date = html_date.attrs['data-date-time']
        except AttributeError:
            news_date = "날짜 없음"
    
        news_companies.append(company)
        news_titles.append(title)
        news_contents.append(content)
        news_dates.append(news_date)
        time.sleep(random.uniform(0.2, 0.4))
    
    # 데이터프레임 생성
    news_df = pd.DataFrame({
        'date': news_dates,
        'title': news_titles,
        'company': news_companies,
        'link': final_urls,
        'content': news_contents
    })
    
    # 중복 제거
    news_df = news_df.drop_duplicates(subset='link', keep='first', ignore_index=True)
    print(f"키워드 '{search}'에 대한 뉴스 수집 완료: {len(news_df)}개 기사 수집")
    

    print(news_df['title'])
    # ====== 키워드 관련성 판단 ======
    # 1. 반도체 관련성 진단
    def is_related_to_semiconductor(title):
        """
        기사 제목이 '반도체'와 관련이 있는지 판단하는 함수.
        관련이 있으면 '반도체', 없으면 '관련 없음' 반환.
        """
        prompt = f"""
        아래는 기사 제목입니다. 이 제목이 '반도체'와 관련이 있는지 판단해 주세요.
        관련이 있으면 '반도체', 관련이 없으면 '관련 없음'이라고 답해 주세요.

        기사 제목:
        {title}

        결과를 둘 중에 한 단어로만 작성해 주세요: '반도체' 또는 '관련 없음'.
        """

        try:
            # OpenAI API 호출
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 기사 제목이 '반도체'와 관련이 있는지 판단하는 도우미입니다."},
                    {"role": "user", "content": prompt}
                ]
            )
            # 응답 추출
            answer = response.choices[0].message.content.strip()

            # 디버깅용 출력
            print(f"제목: {title}")
            print(f"GPT 응답: {answer}\n")

            # 응답 반환
            if answer in ['반도체', '관련 없음']:
                return answer
            else:
                return '관련 없음'  # 예상치 못한 응답 처리
        except Exception as e:
            # 에러 발생 시 처리
            print(f"예외 발생: {e}")
            return '관련 없음'

    # 기사 제목 처리 함수
    def filter_semiconductor_related_articles(news_df):
        """
        DataFrame에서 '반도체'와 관련 있는 기사만 필터링.
        """
        # '관련성' 열 추가
        news_df['관련성'] = news_df['title'].apply(is_related_to_semiconductor)

        # '반도체'와 관련 있는 행만 필터링
        filtered_df = news_df[news_df['관련성'] == '반도체'].reset_index(drop=True)

        return filtered_df

    # 사용 예시
    # DataFrame에서 각 제목에 대해 관련성 판단
    news_df = filter_semiconductor_related_articles(news_df)
    # 결과 확인
    print(news_df[['title', '관련성']])




        
    # 2. 키워드 분류 프로세스
    def get_related_keywords(title, keyword_df):
        # 키워드 목록을 딕셔너리 리스트로 변환
        keywords_list = keyword_df.to_dict('records')  # 각 아이템은 {'키워드': ...} 형태의 딕셔너리

        # 키워드 문자열 생성
        keywords_str = '\n'.join([f"{item['키워드']}" for item in keywords_list])

        prompt = f"""
        아래는 기사 제목과 키워드 목록입니다. 기사 제목이 키워드 목록 중 어떤 키워드와 관련이 있는지 판단하고,
        관련된 키워드를 반드시 키워드 목록(반도체 키워드는 제외) 중 딱 하나로만 알려주세요.
        키워드 목록과 관련 없는 기사 제목은 "관련 없음"이라고 답해주세요.

        기사 제목:
        {title}

        키워드 목록:
        {keywords_str}

        결과를 한 단어(키워드 또는 '관련 없음')로만 작성해주세요.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 기사 제목과 키워드 목록을 비교하여 관련된 키워드를 찾아주는 도우미입니다."},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.choices[0].message.content.strip()
            
            # 디버깅을 위해 GPT의 응답을 출력합니다.
            print(f"제목: {title}")
            print(f"GPT 응답: {answer}\n")
            
            if answer == '관련 없음':
                return None
            else:
                return answer
        except Exception as e:
            # 예외 메시지를 출력합니다.
            print(f"예외 발생: {e}")
            return None
    
    def process_article(title):
        keyword = get_related_keywords(title, keyword_df)
        if keyword is None:
            return "관련 없음"  # 키워드가 없으면 기본값 반환
        return keyword

        
    # 함수 적용하여 '구분'과 '키워드' 칼럼 추가
    print(f"키워드 '{search}'에 대한중복내용 제거")
    news_df['키워드'] = news_df['title'].apply(process_article)
    
    # 관련 없는 기사 제거

    news_df = news_df[news_df['키워드'].notna()].reset_index(drop=True)
    print(news_df['title'])

    # 중복 제거 및 그룹별 데이터프레임 반환
    
    # 3. 제목유사도 통한 중복제거
    # GPT를 사용하여 제목 유사도 판단
    def group_similar_titles(titles):
        """
        여러 뉴스 제목을 GPT 모델에게 전달해 같은 뜻의 제목끼리 그룹화합니다.

        Parameters:
        - titles (list): 뉴스 제목 리스트

        Returns:
        - grouped_titles (dict): 유사한 제목 그룹 딕셔너리
        """
        # 프롬프트 생성
        prompt = f"""
        다음 뉴스 제목들을 서로 비교하여, 완전히 같은 뜻의 제목끼리 그룹화해주세요.
        완전히 같은 의미의 제목끼리 묶어 한번에 다루려고 합니다.
        동일한 그룹에 속하는 제목끼리는 순서대로 나열해 주세요. 서로 다른 그룹은 번호를 매겨 구분해 주세요.
        최대한 그룹 수가 많게 해주세요

        제목 리스트:
        {chr(10).join([f"{i + 1}. {title}" for i, title in enumerate(titles)])}

        결과는 다음과 같은 형식으로 출력해 주세요:
        - 그룹 1: 제목 번호, 제목 번호
        - 그룹 2: 제목 번호
        - 그룹 3: 제목 번호, 제목 번호        
        ...
        """
        
        try:
            # GPT 모델 호출
            messages = [
                {"role": "system", "content": "뉴스 제목의 유사성을 판단하여 그룹화하는 도우미입니다."},
                {"role": "user", "content": prompt}
            ]
            completion = client.chat.completions.create(
                model="gpt-4o-mini",  # 또는 "gpt-3.5-turbo"
                messages=messages
            )
            # GPT 응답 처리
            response = completion.choices[0].message.content.strip()
            
            # 응답 파싱
            grouped_titles = {}
            for line in response.split("\n"):
                if line.startswith("- 그룹"):  # "그룹"으로 시작하는 줄만 처리
                    try:
                        group, members = line.split(": ")
                        group_id = int(re.search(r"\d+", group).group())  # 그룹 번호 추출
                        member_ids = [int(re.search(r"\d+", x).group()) for x in members.split(",")]
                        grouped_titles[group_id] = member_ids
                    except (ValueError, AttributeError) as ve:
                        print(f"파싱 에러 발생: {ve} / 문제 있는 줄: {line}")
            return grouped_titles

        except Exception as e:
            print(f"에러 발생: {e}")
            return {}

        
    def remove_duplicates_by_group(df):
        # 데이터프레임의 인덱스를 리셋 (0-based로 강제)
        df = df.reset_index(drop=True)
        
        titles = df["title"].tolist()
        groups = group_similar_titles(titles)  # 유사한 제목 그룹화
        
        # 그룹화 결과 출력
        print("\n[그룹화 결과]")
        for group, indices in groups.items():
            print(f"그룹 {group}: {indices}")
        
        to_remove = set()

        for group, indices in groups.items():
            # 그룹 내 제목 비교
            # GPT의 응답이 1-based라 가정하고 0-based로 변환
            group_indices = sorted([idx - 1 for idx in indices])  # -1 오프셋 적용
            
            for i in range(len(group_indices)):
                if group_indices[i] in to_remove:
                    continue
                for j in range(i + 1, len(group_indices)):
                    if group_indices[j] in to_remove:
                        continue
                    # 본문 길이 비교
                    idx_i, idx_j = group_indices[i], group_indices[j]
                    if len(df.loc[idx_i, "content"]) >= len(df.loc[idx_j, "content"]):
                        to_remove.add(idx_j)  # 짧은 본문 제거
                    else:
                        to_remove.add(idx_i)  # 현재 i 제거
                        break

        # 중복 제거 후 데이터프레임 반환
        return df.drop(index=list(to_remove)).reset_index(drop=True)

    news_df_filtered = remove_duplicates_by_group(news_df)

    print(news_df_filtered['title'])
    
    print(f"키워드 '{search}'에 대한 중복제거 완료: 최종 {len(news_df_filtered)}개 기사 ")
    
    # ====== 기사 요약 ======
    def summarize_content(text, max_tokens=1000):
        try:
            if not text:
                raise ValueError("기사 내용이 비어 있습니다.")
            
            prompt = f"다음 기사를 200자 이내로 요약해주세요:\n\n{text}"
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 뉴스 기사를 요약하는 도우미입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            # print(f"요약 중 오류 발생: {e}")
            return "요약 실패"
    
    news_df_filtered['summary'] = news_df_filtered['content'].apply(lambda x: summarize_content(x) if len(x) > 10 else "내용 부족")
    
    # '검색어' 칼럼 추가
    news_df_filtered['검색어'] = search
    
    # 수집된 뉴스 데이터를 전체 DataFrame에 추가
    all_news_df = pd.concat([all_news_df, news_df_filtered], ignore_index=True)
    
    print(f"키워드 '{search}'에 대한 데이터 처리 완료\n")

# 전체 수집 완료 후, 전체 데이터를 저장

# 중복 제거 한번더
all_news_df = all_news_df.drop_duplicates(subset='link', keep='first', ignore_index=True)

def group_similar_titles(titles):
        """
        여러 뉴스 제목을 GPT 모델에게 전달해 유사한 제목끼리 그룹화합니다.

        Parameters:
        - titles (list): 뉴스 제목 리스트

        Returns:
        - grouped_titles (dict): 유사한 제목 그룹 딕셔너리
        """
        # 프롬프트 생성
        prompt = f"""
        다음 뉴스 제목들을 서로 비교하여, 완전히 같은 뜻의 제목끼리 그룹화해주세요.
        완전히 같은 뜻 제목만 그룹화하면 됩니다.

        동일한 그룹에 속하는 제목끼리는 순서대로 나열해 주세요. 서로 다른 그룹은 번호를 매겨 구분해 주세요.
        최대한 그룹 수가 많게 해주세요

        제목 리스트:
        {chr(10).join([f"{i + 1}. {title}" for i, title in enumerate(titles)])}

        결과는 다음과 같은 형식으로 출력해 주세요:
        - 그룹 1: 제목 번호, 제목 번호
        - 그룹 2: 제목 번호
        - 그룹 3: 제목 번호, 제목 번호
        ...
        """
        
        try:
            # GPT 모델 호출
            messages = [
                {"role": "system", "content": "뉴스 제목 리스트를 보고 같은 뜻의 뉴스 제목을 그룹화하는 도우미입니다."},
                {"role": "user", "content": prompt}
            ]
            completion = client.chat.completions.create(
                model="gpt-4o-mini",  # 또는 "gpt-3.5-turbo"
                messages=messages
            )
            # GPT 응답 처리
            response = completion.choices[0].message.content.strip()
            
            # 응답 파싱
            grouped_titles = {}
            for line in response.split("\n"):
                if line.startswith("- 그룹"):  # "그룹"으로 시작하는 줄만 처리
                    try:
                        group, members = line.split(": ")
                        group_id = int(re.search(r"\d+", group).group())  # 그룹 번호 추출
                        member_ids = [int(re.search(r"\d+", x).group()) for x in members.split(",")]
                        grouped_titles[group_id] = member_ids
                    except (ValueError, AttributeError) as ve:
                        print(f"파싱 에러 발생: {ve} / 문제 있는 줄: {line}")
            return grouped_titles

        except Exception as e:
            print(f"에러 발생: {e}")
            return {}

        
def remove_duplicates_by_group(df):
    # 데이터프레임의 인덱스를 리셋 (0-based로 강제)
    df = df.reset_index(drop=True)
    
    titles = df["title"].tolist()
    groups = group_similar_titles(titles)  # 유사한 제목 그룹화
    
    # 그룹화 결과 출력
    print("\n[그룹화 결과]")
    for group, indices in groups.items():
        print(f"그룹 {group}: {indices}")
    
    to_remove = set()

    for group, indices in groups.items():
        # 그룹 내 제목 비교
        # GPT의 응답이 1-based라 가정하고 0-based로 변환
        group_indices = sorted([idx - 1 for idx in indices])  # -1 오프셋 적용
        
        for i in range(len(group_indices)):
            if group_indices[i] in to_remove:
                continue
            for j in range(i + 1, len(group_indices)):
                if group_indices[j] in to_remove:
                    continue
                # 본문 길이 비교
                idx_i, idx_j = group_indices[i], group_indices[j]
                if len(df.loc[idx_i, "content"]) >= len(df.loc[idx_j, "content"]):
                    to_remove.add(idx_j)  # 짧은 본문 제거
                else:
                    to_remove.add(idx_i)  # 현재 i 제거
                    break

    # 중복 제거 후 데이터프레임 반환
    return df.drop(index=list(to_remove)).reset_index(drop=True)
print(all_news_df['title'])
all_news_df_filtered = remove_duplicates_by_group(all_news_df)
print(all_news_df_filtered['title'])


now = datetime.datetime.now()
file_name = f"Total_Filtered_No_Comment.csv"
existing_data = load_existing_data(file_name)

all_news_df_filtered['date'] = pd.to_datetime(all_news_df_filtered['date'], errors='coerce').dt.strftime('%Y-%m-%d')
# 기존 데이터와 새 데이터 병합
updated_data = merge_and_remove_duplicates(existing_data, all_news_df_filtered)

# 병합된 데이터 저장
save_updated_data(updated_data, file_name)
