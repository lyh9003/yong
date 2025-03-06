import streamlit as st
import pandas as pd
import datetime

GITHUB_CSV_URL = "https://raw.githubusercontent.com/lyh9003/yong/main/Total_Filtered_No_Comment.csv"

st.return()

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
    #    → 하나의 기사에 키워드가 여러 개인 경우, 여러 행으로 복제
    df = df.explode('키워드_목록', ignore_index=True)
    
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
st.title("📢 반도체 뉴스 업데이트")
st.write("yh9003.lee@samsung.com")
# ======================================================
# 1) 사이드바에서 날짜를 여러 개 선택할 수 있는 필터
# ======================================================
# 날짜만 추출해서 중복 제거 후 내림차순 정렬
unique_dates = sorted(list(set(df['date'].dt.date.dropna())), reverse=True)

selected_dates = st.sidebar.multiselect(
    "날짜를 선택하세요 (복수 선택 가능)",
    unique_dates,
    help="아무 것도 선택하지 않으면 최근 1주일치 기사가 표시됩니다."
)

# 날짜 필터 적용
if selected_dates:
    display_df = df[df['date'].dt.date.isin(selected_dates)]
else:
    display_df = default_recent_df

st.write(f"**총 기사 수:** {len(display_df)}개")

# ======================================================
# 2) 날짜별 → 키워드별 → 기사 목록 표시
# ======================================================
# 날짜(date)만으로 그룹핑(내림차순 유지)
grouped_by_date = display_df.groupby(display_df['date'].dt.date, sort=False)

for current_date, date_group in grouped_by_date:
    st.markdown(f"## {current_date.strftime('%Y-%m-%d')}")

    # 키워드 기준으로 다시 그룹핑 (날짜 그룹 내부)
    grouped_by_keyword = date_group.groupby('키워드_목록', sort=False)
    
    for keyword_value, keyword_group in grouped_by_keyword:
        # 키워드가 비어있지 않은 경우에만 표시
        if pd.notna(keyword_value) and str(keyword_value).strip():
            st.markdown(f"### ▶️ {keyword_value}")
        else:
            # 빈 키워드 그룹 처리
            st.markdown("### ▶️ (키워드 없음)")
        
        # 해당 키워드에 속한 기사들 표시
        for idx, row in keyword_group.iterrows():
            with st.expander(f"📰 {row['title']}"):
                # 요약 보기 버튼
                if st.button(f"요약 보기: {row['title']}", key=f"summary_{idx}"):
                    st.write(row.get('summary', '요약 정보가 없습니다.'))

                # 기사 링크
                link = row.get('link', None)
                if pd.notna(link):
                    st.markdown(f"[🔗 기사 링크]({link})")
                else:
                    st.write("링크가 없습니다.")
