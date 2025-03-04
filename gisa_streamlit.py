import streamlit as st
import pandas as pd

# GitHub Raw CSV 파일 URL
GITHUB_CSV_URL = "https://raw.githubusercontent.com/lyh9003/yong/main/Total_Filtered_No_Comment.csv"

@st.cache_data
def load_data():
    """CSV를 불러와 DataFrame으로 반환합니다."""
    df = pd.read_csv(GITHUB_CSV_URL, encoding='utf-8-sig')
    # 날짜 컬럼을 datetime으로 변환
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # 내림차순 정렬
    df = df.sort_values(by='date', ascending=False)
    
    # 키워드 컬럼을 리스트로 분리 (',' 기준)
    # keywords_list 컬럼을 새로 만들어서 사용할 예정
    def split_keywords(k):
        # NaN 방지용으로 str 처리 후 split
        if pd.isna(k):
            return []
        return [x.strip() for x in str(k).split(',') if x.strip()]
    
    df['keywords_list'] = df['keywords'].apply(split_keywords)
    
    return df

# 데이터 불러오기
df = load_data()

# Streamlit 앱 타이틀
st.title("📢 반도체 뉴스 탐색")

# =================================
# 1) 키워드 필터링 사이드바
# =================================
unique_keywords = sorted(set(sum(df['keywords_list'], [])))  # 모든 키워드를 합쳐서 unique하게 정렬
selected_keywords = st.sidebar.multiselect("키워드로 기사 필터링", unique_keywords, help="관심 있는 키워드를 선택하세요.")

if selected_keywords:
    # 선택한 키워드를 포함하는 기사만 필터링
    filtered_df = df[df['keywords_list'].apply(lambda x: any(k in x for k in selected_keywords))]
else:
    filtered_df = df.copy()

st.write(f"선택한 키워드가 들어간 기사 수: **{len(filtered_df)}개**")

# =================================
# 2) 날짜별 주요 키워드 및 기사 목록
# =================================
# date 컬럼을 Date만 추출해서 그룹핑 (시계열이 아닌 날짜 단위로 그룹화)
grouped = filtered_df.groupby(filtered_df['date'].dt.date, sort=False)

for current_date, group_data in grouped:
    # 날짜 헤더
    st.markdown(f"## {current_date.strftime('%Y-%m-%d')}")

    # (1) 날짜 그룹 내 주요 키워드 top 5 계산
    exploded_keywords = group_data['keywords_list'].explode()  # 리스트 형태의 키워드 컬럼을 펼침
    keyword_counts = exploded_keywords.value_counts().head(5)

    if not keyword_counts.empty:
        st.write("**주요 키워드 TOP 5**")
        for kw, cnt in keyword_counts.items():
            st.write(f"- {kw} ({cnt}회)")

    # (2) 날짜 그룹 내 기사 상세 정보
    for idx, row in group_data.iterrows():
        with st.expander(f"📅 {row['date'].strftime('%Y-%m-%d')} - {row['title']}"):
            st.write("**키워드**:", ", ".join(row['keywords_list']))
            
            # 요약 버튼
            if st.button(f"요약 보기: {row['title']}", key=f"summary_{idx}"):
                st.write(row['summary'])
            
            # 본문 버튼
            if st.button(f"본문 보기: {row['title']}", key=f"content_{idx}"):
                st.write(row['content'])
            
            # 기사 링크
            if pd.notna(row['link']):
                st.markdown(f"[🔗 기사 링크]({row['link']})")
            else:
                st.write("링크가 없습니다.")
