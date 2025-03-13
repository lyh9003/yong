import streamlit as st
import pandas as pd
import datetime
import time

# RAG 관련 라이브러리 임포트
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

# =============================================================================
# RAG를 위한 벡터 스토어 생성 함수 (캐싱 처리)
# =============================================================================
@st.cache_resource
def create_vector_store(dataframe):
    # 각 기사에서 제목과 요약을 결합하여 텍스트 문서를 생성합니다.
    texts = []
    for _, row in dataframe.iterrows():
        title = row.get('title', '')
        summary = row.get('summary', '')
        combined_text = f"제목: {title}\n요약: {summary}"
        texts.append(combined_text)
    
    # 문서의 길이가 길 경우를 대비해 텍스트 분할기를 사용합니다.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_texts = []
    for text in texts:
        split_texts.extend(text_splitter.split_text(text))
    
    # OpenAI 임베딩을 사용하여 Chroma 벡터 스토어 생성 (persist_directory를 None으로 하여 메모리 내 저장)
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_texts(split_texts, embedding=embeddings, persist_directory=None)
    return vector_store

# =============================================================================
# 날짜 필터링을 위한 기본 데이터 준비
# =============================================================================
if not df.empty:
    max_date = df['date'].max()
    one_week_ago = max_date - datetime.timedelta(days=7)
    one_month_ago = max_date - datetime.timedelta(days=30)
else:
    one_week_ago = one_month_ago = None

# =============================================================================
# Streamlit 앱 타이틀
# =============================================================================
st.title("📢반도체 뉴스레터(Rev.25.3.13)")
st.write("문의/아이디어 : yh9003.lee@samsung.com")

# =============================================================================
# 사이드바 필터 옵션 설정 (날짜, 키워드, 검색어)
# =============================================================================
date_filter_option = st.sidebar.radio(
    "📅 날짜 필터 옵션",
    ["최근 7일", "최근 1달", "전체", "직접 선택"],
    index=0
)

unique_dates = sorted(list(set(df['date'].dt.date.dropna())), reverse=True)

if date_filter_option == "최근 7일":
    selected_dates = [date for date in unique_dates if date >= one_week_ago.date()]
elif date_filter_option == "최근 1달":
    selected_dates = [date for date in unique_dates if date >= one_month_ago.date()]
elif date_filter_option == "전체":
    selected_dates = unique_dates
else:  # "직접 선택"
    selected_dates = st.sidebar.multiselect(
        "📅 날짜를 선택하세요 (복수 선택 가능)",
        unique_dates,
        help="필터 옵션에서 '직접 선택'을 선택한 경우에만 활성화됩니다."
    )

unique_keywords = sorted(list(df['키워드_목록'].dropna().unique()))
selected_keywords = st.sidebar.multiselect(
    "🔍 키워드를 선택하세요 (복수 선택 가능)",
    unique_keywords,
    help="아무 것도 선택하지 않으면 모든 키워드가 표시됩니다."
)

search_query = st.sidebar.text_input(
    "🔎 검색어 입력 (제목/요약 포함)",
    help="특정 단어가 포함된 기사만 검색합니다."
)

# =============================================================================
# 필터 적용 (날짜 + 키워드 + 검색어)
# =============================================================================
filtered_df = df.copy()

if selected_dates:
    filtered_df = filtered_df[filtered_df['date'].dt.date.isin(selected_dates)]

if selected_keywords:
    filtered_df = filtered_df[filtered_df['키워드_목록'].isin(selected_keywords)]

if search_query:
    search_query_lower = search_query.lower()
    filtered_df = filtered_df[
        filtered_df['title'].str.lower().str.contains(search_query_lower, na=False) |
        filtered_df['summary'].fillna('').str.lower().str.contains(search_query_lower, na=False)
    ]

st.write(f"**총 기사 수:** {len(filtered_df)}개")

# =============================================================================
# 날짜별 → 키워드별 → 기사 목록 표시 (제목 클릭 시 요약 & 링크 표시)
# =============================================================================
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
            with st.expander(f"📰 {row['title']}"):
                st.write(f"**요약:** {row.get('summary', '요약 정보가 없습니다.')}")
                link = row.get('link', None)
                if pd.notna(link):
                    st.markdown(f"[🔗 기사 링크]({link})")
                else:
                    st.write("링크가 없습니다.")

# =============================================================================
# RAG (Retrieval-Augmented Generation) 기능 추가
# =============================================================================
st.markdown("## 🤖 RAG 질문하기")
rag_question = st.text_input("기사 내용에 대해 질문을 입력하세요", key="rag_question")

if rag_question:
    with st.spinner("답변 생성 중..."):
        # 전체 기사 데이터를 기반으로 벡터 스토어 생성 (캐싱되어 있음)
        vector_store = create_vector_store(df)
        
        # 사용자의 질문과 유사한 문서 조각 3개를 검색합니다.
        retrieved_docs = vector_store.similarity_search(rag_question, k=3)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # ChatGPT에 전달할 프롬프트 구성 (문맥 정보 + 질문)
        prompt = f"다음 기사 정보들을 참고하여 질문에 답변하세요:\n\n{context}\n\n질문: {rag_question}\n답변:"
        
        chat = ChatOpenAI(temperature=0.7)
        response = chat([{"role": "user", "content": prompt}])
        answer = response.content
        
    st.markdown("### 답변")
    st.write(answer)
