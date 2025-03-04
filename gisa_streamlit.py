import streamlit as st
import pandas as pd

# GitHub의 Raw CSV 파일 URL
GITHUB_CSV_URL = "https://raw.githubusercontent.com/lyh9003/yong/main/Total_Filtered_No_Comment.csv"

@st.cache_data
def load_data():
    return pd.read_csv(GITHUB_CSV_URL, encoding='utf-8-sig')

# 데이터 불러오기
df = load_data()

# 날짜 컬럼을 datetime 형식으로 변환 후 내림차순 정렬 (컬럼명 맞춰 수정 필요)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date', ascending=False)

# Streamlit 앱 시작
st.title("📢 반도체 뉴스 탐색")

for index, row in df.iterrows():
    with st.expander(f"📅 {row['date'].strftime('%Y-%m-%d')} - {row['title']}"):
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button(f"요약 보기: {row['title']}", key=f"summary_{index}"):
                st.write(row['summary'])
        
        with col2:
            if st.button(f"본문 보기: {row['title']}", key=f"content_{index}"):
                st.write(row['content'])
        
        with col3:
            st.markdown(f"[🔗 기사 링크]({row['link']})")
