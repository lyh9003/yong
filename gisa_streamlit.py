import streamlit as st
import pandas as pd
import datetime
import time

GITHUB_CSV_URL = f"https://raw.githubusercontent.com/lyh9003/yong/main/Total_Filtered_No_Comment.csv?nocache={int(time.time())}"

def load_data():
    """CSV를 불러와 DataFrame으로 반환합니다."""
    df = pd.read_csv(GITHUB_CSV_URL, encoding='utf-8-sig')
    
    # 1) 날짜 컬럼을 datetime 형식으로 변환
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # 2) 내림차순 정렬 (가장 최근 기사가 위로)
    df = df.sort_values(by='date', ascending=False)
    
    # 3) '키워드' 컬럼을 쉼표 기준으로 분할하여 리스트화
    def split_keywords(kw_string):
        if pd.isna(kw_string):
            return []
        return [k.strip() for k in kw_string.split(',') if k.strip()]
    
    df['키워드_목록'] = df['키워드'].apply(split_keywords)
    
    # 4) explode를 이용하여 키워드별로 레코드를 펼침
    df = df.explode('키워드_목록', ignore_index=True)
    
    # 5) '관련 없음'을 '기타'로 변경
    df['키워드_목록'] = df['키워드_목록'].replace('관련 없음', '기타')
    
    return df

# 데이터 불러오기
df = load_data()

# ======================================================
# 0) 최근 1주일치 필터링을 위한 기본 데이터 준비
# ======================================================
if not df.empty:
    max_date = df['date'].max()
    one_week_ago = max_date - datetime.timedelta(days=7)
    default_recent_df = df[df['date'] >= one_week_ago]
else:
    default_recent_df = df.copy()

# ======================================================
# Streamlit 앱 타이틀
# ======================================================
st.title("📢 반도체 뉴스레터(Rev/25.3.13)")
st.write("yh9003.lee@samsung.com")

# ======================================================
# 1) 사이드바 필터 (날짜 선택)
# ======================================================
unique_dates = sorted(list(set(df['date'].dt.date.dropna())), reverse=True)

selected_dates = st.sidebar.multiselect(
    "📅 날짜를 선택하세요 (복수 선택 가능)",
    unique_dates,
    help="아무 것도 선택하지 않으면 최근 1주일치 기사가 표시됩니다."
)

# ======================================================
# 2) 키워드 필터 추가 (카테고리 역할, '관련 없음' → '기타')
# ======================================================
unique_keywords = sorted(list(df['키워드_목록'].dropna().unique()))

selected_keywords = st.sidebar.multiselect(
    "🔍 키워드를 선택하세요 (복수 선택 가능)",
    unique_keywords,
    help="아무 것도 선택하지 않으면 모든 키워드가 표시됩니다."
)

# ======================================================
# 3) 검색어 필터 추가 (제목 및 요약 검색)
# ======================================================
search_query = st.sidebar.text_input(
    "🔎 검색어 입력 (제목/요약 포함)",
    help="특정 단어가 포함된 기사만 검색합니다."
)

# ======================================================
# 4) 필터 적용 (날짜 + 키워드 + 검색어)
# ======================================================
filtered_df = df.copy()

# 날짜 필터 적용
if selected_dates:
    filtered_df = filtered_df[filtered_df['date'].dt.date.isin(selected_dates)]
else:
    filtered_df = default_recent_df

# 키워드 필터 적용
if selected_keywords:
    filtered_df = filtered_df[filtered_df['키워드_목록'].isin(selected_keywords)]

# 검색어 필터 적용
if search_query:
    search_query = search_query.lower()
    filtered_df = filtered_df[
        filtered_df['title'].str.lower().str.contains(search_query, na=False) |
        filtered_df['summary'].fillna('').str.lower().str.contains(search_query, na=False)
    ]


st.write(f"**총 기사 수:** {len(filtered_df)}개")

# ======================================================
# 5) 날짜별 → 키워드별 → 기사 목록 표시
# ======================================================
grouped_by_date = filtered_df.groupby(filtered_df['date'].dt.date, sort=False)

for current_date, date_group in grouped_by_date:
    st.markdown(f"## {current_date.strftime('%Y-%m-%d')}")

    grouped_by_keyword = date_group.groupby('키워드_목록', sort=False)
    
    for keyword_value, keyword_group in grouped_by_keyword:
        if pd.notna(keyword_value) and str(keyword_value).strip():
            st.markdown(f"### ▶️ {keyword_value}")
        else:
            st.markdown("### ▶️ (키워드 없음)")
        
        for idx, row in keyword_group.iterrows():
            # 제목을 버튼으로 만들어 클릭 시 요약이 표시되도록 함
            if st.button(f"📰 {row['title']}", key=f"title_{idx}"):
                st.write(f"**요약:** {row.get('summary', '요약 정보가 없습니다.')}")
                
                link = row.get('link', None)
                if pd.notna(link):
                    st.markdown(f"[🔗 기사 링크]({link})")
                else:
                    st.write("링크가 없습니다.")
