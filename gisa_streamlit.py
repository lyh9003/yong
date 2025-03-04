import streamlit as st
import pandas as pd
import datetime

GITHUB_CSV_URL = "https://raw.githubusercontent.com/lyh9003/yong/main/Total_Filtered_No_Comment.csv"

@st.cache_data
def load_data():
    """CSV를 불러와 DataFrame으로 반환합니다."""
    df = pd.read_csv(GITHUB_CSV_URL, encoding='utf-8-sig')
    
    # 날짜 컬럼을 datetime으로 변환 (에러 발생 시 NaT 처리)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # 내림차순 정렬
    df = df.sort_values(by='date', ascending=False)
    
    return df

# 데이터 불러오기
df = load_data()

# -- 최근 1주일치 데이터 필터링
if not df.empty:
    # 데이터에 있는 'date' 중 최댓값 (가장 최근 날짜) 구하기
    max_date = df['date'].max()
    # 최근 7일 전
    one_week_ago = max_date - datetime.timedelta(days=7)
    
    # 최근 1주일치 데이터
    default_recent_df = df[df['date'] >= one_week_ago]
else:
    # 만약 df가 비어있다면 빈 데이터프레임 할당
    default_recent_df = df.copy()

# Streamlit 앱 타이틀
st.title("📢 반도체 뉴스 탐색 (최근 1주일치 + 날짜 필터)")

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
    # 사용자가 날짜를 선택한 경우 → 그 날짜만 보여주기
    display_df = df[df['date'].dt.date.isin(selected_dates)]
else:
    # 아무 것도 선택 안 한 경우 → 최근 1주일치
    display_df = default_recent_df

st.write(f"**총 기사 수:** {len(display_df)}개")

# ======================================================
# 2) 기사 표시 (날짜별로 그룹화)
# ======================================================
# 날짜만으로 그룹핑(내림차순)
grouped = display_df.groupby(display_df['date'].dt.date, sort=False)

for current_date, group_data in grouped:
    st.markdown(f"## {current_date.strftime('%Y-%m-%d')}")
    for idx, row in group_data.iterrows():
        # 날짜 + 제목으로 expander 표시
        with st.expander(f"📰 {row['title']}"):
            # (1) 요약 보기 버튼
            if st.button(f"요약 보기: {row['title']}", key=f"summary_{idx}"):
                st.write(row.get('summary', '요약 정보가 없습니다.'))

            # (3) 기사 링크
            link = row.get('link', None)
            if pd.notna(link):
                st.markdown(f"[🔗 기사 링크]({link})")
            else:
                st.write("링크가 없습니다.")
