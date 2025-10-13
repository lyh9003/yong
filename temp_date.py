import pandas as pd
from bs4 import BeautifulSoup
import requests
import time
import random
from tqdm import tqdm

def get_date_from_url(url):
    """
    네이버 뉴스 URL에서 날짜를 크롤링하는 함수
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://www.naver.com/"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 날짜 추출 (업데이트된 셀렉터 사용)
        html_date = soup.select_one("span.media_end_head_info_datestamp_time")
        
        if html_date and 'data-date-time' in html_date.attrs:
            date_time = html_date['data-date-time']
            # 날짜 부분만 추출 (YYYY-MM-DD 형식)
            return date_time.split()[0]
        else:
            # 대체 셀렉터 시도
            html_date = soup.select_one("span[data-date-time]")
            if html_date:
                date_time = html_date['data-date-time']
                return date_time.split()[0]
            else:
                return None
                
    except Exception as e:
        print(f"오류 발생 ({url}): {e}")
        return None

def fill_missing_dates(csv_file):
    """
    CSV 파일에서 누락된 날짜를 채우는 함수
    """
    # CSV 파일 읽기
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    
    # 날짜가 누락된 행 찾기
    missing_date_mask = (
        df['date'].isna() | 
        (df['date'] == '날짜 없음') | 
        (df['date'] == '') |
        (df['date'] == 'NaT')
    )
    
    missing_count = missing_date_mask.sum()
    print(f"\n날짜가 누락된 행: {missing_count}개")
    
    if missing_count == 0:
        print("누락된 날짜가 없습니다!")
        return df
    
    # 누락된 날짜를 가진 행들에 대해 크롤링
    print("\n날짜 크롤링 시작...")
    
    for idx in tqdm(df[missing_date_mask].index):
        url = df.loc[idx, 'link']
        
        if pd.isna(url) or url == '':
            print(f"\n{idx}번 행: URL이 없습니다.")
            continue
        
        # 날짜 크롤링
        date = get_date_from_url(url)
        
        if date:
            df.loc[idx, 'date'] = date
            print(f"\n{idx}번 행 업데이트: {date}")
        else:
            print(f"\n{idx}번 행: 날짜를 찾을 수 없습니다.")
        
        # 서버 부하 방지를 위한 딜레이
        time.sleep(random.uniform(0.3, 0.6))
    
    return df

# 실행
if __name__ == "__main__":
    csv_file = "Total_Filtered_No_Comment.csv"
    
    print(f"CSV 파일 로드: {csv_file}")
    
    # 날짜 채우기
    updated_df = fill_missing_dates(csv_file)
    
    # 결과 저장
    output_file = "Total_Filtered_No_Comment_Updated.csv"
    updated_df.to_csv(output_file, encoding='utf-8-sig', index=False)
    
    print(f"\n✅ 업데이트 완료! 저장된 파일: {output_file}")
    
    # 결과 확인
    remaining_missing = (
        updated_df['date'].isna() | 
        (updated_df['date'] == '날짜 없음') | 
        (updated_df['date'] == '')
    ).sum()
    
    print(f"남은 누락된 날짜: {remaining_missing}개")
