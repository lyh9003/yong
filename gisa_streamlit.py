import streamlit as st
import pandas as pd
import datetime
import time

# RAG 관련 라이브러리 임포트
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

# CSV 파일 URL (GitHub)
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
# 날짜 필터링을 위한 기본 데이터 준비
# ======================================================
if not df.empty:
    max_date = df['date'].max()
    one_week_ago = max_date - datetime.timedelta(days=7)
    one_month_ago = max_date - datetime.timedelta(days=30)
else:
    one_week_ago = one_month_ago = None

# ======================================================
# Streamlit 앱 타이틀
# ======================================================
st.title("📢반도체 뉴스레터(Rev.25.3.13)")
st.write("문의/아이디어 : yh9003.lee@samsung.com")

# ======================================================
# 사이드바 날짜 필터 옵션 추가
# ======================================================
date_filter_option = st.sidebar.radio(
    "📅 날짜 필터 옵션",
    ["최근 7일", "최근 1달", "전체", "직접 선택"],
    index=0
)

unique_dates = sorted(list(set(df['date'].dt.date.dropna())), reverse=True)

# 날짜 필터 옵션 적용
if date_filter_option == "최근 7일":
    selected_dates = [date for date in unique_dates if date >= one_week_ago.date()]
elif date_filter_option == "최근 1달":
    selected_dates = [date for date in unique_dates if date >= one_month_ago.date()]
elif date_filter_option == "전체":
    selected_dates = unique_dates  # 모든 날짜 포함
else:  # "직접 선택"
    selected_dates = st.sidebar.multiselect(
        "📅 날짜를 선택하세요 (복수 선택 가능)",
        unique_dates,
        help="필터 옵션에서 '직접 선택'을 선택한 경우에만 활성화됩니다."
    )

# ======================================================
# 키워드 필터 추가 (카테고리 역할, '관련 없음' → '기타')
# ======================================================
unique_keywords = sorted(list(df['키워드_목록'].dropna().unique()))

selected_keywords = st.sidebar.multiselect(
    "🔍 키워드를 선택하세요 (복수 선택 가능)",
    unique_keywords,
    help="아무 것도 선택하지 않으면 모든 키워드가 표시됩니다."
)

# ======================================================
# 검색어 필터 추가 (제목 및 요약 검색)
# ======================================================
search_query = st.sidebar.text_input(
    "🔎 검색어 입력 (제목/요약 포함)",
    help="특정 단어가 포함된 기사만 검색합니다."
)

# ======================================================
# 필터 적용 (날짜 + 키워드 + 검색어)
# ======================================================
filtered_df = df.copy()

# 날짜 필터 적용
if selected_dates:
    filtered_df = filtered_df[filtered_df['date'].dt.date.isin(selected_dates)]

# 키워드 필터 적용
if selected_keywords:
    filtered_df = filtered_df[filtered_df['키워드_목록'].isin(selected_keywords)]

# 검색어 필터 적용 (제목 + 요약)
if search_query:
    search_query = search_query.lower()
    filtered_df = filtered_df[
        filtered_df['title'].str.lower().str.contains(search_query, na=False) |
        filtered_df['summary'].fillna('').str.lower().str.contains(search_query, na=False)
    ]

st.write(f"**총 기사 수:** {len(filtered_df)}개")

# ======================================================
# 날짜별 → 키워드별 → 기사 목록 표시 (제목 클릭 시 요약 & 링크 표시)
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
            with st.expander(f"📰 {row['title']}"):
                st.write(f"**요약:** {row.get('summary', '요약 정보가 없습니다.')}")
                link = row.get('link', None)
                if pd.notna(link):
                    st.markdown(f"[🔗 기사 링크]({link})")
                else:
                    st.write("링크가 없습니다.")

# ======================================================
# RAG(검색기반 생성) 기능 구현
# ======================================================

# 캐싱을 통해 벡터스토어 빌드 비용 최소화 (Streamlit 1.18 이상 st.cache_resource 사용)
# RAG용 벡터스토어 생성 함수 내 수정 예시
@st.cache_resource
def build_vectorstore(dataframe: pd.DataFrame):
    """
    DataFrame의 각 기사에서 제목과 요약을 결합한 텍스트를 생성하고,
    SemanticChunker로 청크를 만든 후 FAISS 벡터스토어를 생성합니다.
    """
    documents = []
    metadatas = []
    for _, row in dataframe.iterrows():
        # 제목과 요약을 결합 (요약이 없으면 제목만 사용)
        text = row.get('title', '')
        summary = row.get('summary', '')
        if pd.notna(summary) and summary.strip():
            text += "\n" + summary
        documents.append(text)
        metadatas.append({
            "date": row.get("date"),
            "키워드": row.get("키워드_목록")
        })
    
    # LLM 인스턴스를 chunker용으로 생성 (온도나 기타 파라미터는 필요에 맞게 조정)
    llm_for_chunking = Ollama(model="llama2", temperature=0)
    # 필요한 인자를 전달하여 SemanticChunker 초기화 (예: chunk_size, chunk_overlap 등)
    chunker = SemanticChunker(llm=llm_for_chunking, chunk_size=512, chunk_overlap=50)
    
    docs_chunks = []
    docs_metadatas = []
    for doc, meta in zip(documents, metadatas):
        chunks = chunker.split_text(doc)
        docs_chunks.extend(chunks)
        docs_metadatas.extend([meta] * len(chunks))
    
    # 임베딩 계산 (HuggingFaceEmbeddings 사용)
    embeddings = HuggingFaceEmbeddings()
    
    # FAISS 벡터스토어 생성
    vectorstore = FAISS.from_texts(docs_chunks, embeddings, metadatas=docs_metadatas)
    return vectorstore

def answer_query(query: str, vectorstore, top_k: int = 3) -> str:
    """
    입력된 질의에 대해 벡터스토어에서 유사한 청크들을 검색하고,
    검색된 문맥을 바탕으로 Ollama LLM이 답변을 생성합니다.
    """
    # 관련 문서 검색
    docs = vectorstore.similarity_search(query, k=top_k)
    context = "\n".join([doc.page_content for doc in docs])
    
    # LLM에 전달할 프롬프트 구성
    prompt = (
        f"다음 문맥을 참고하여 질문에 답변하세요:\n\n"
        f"문맥:\n{context}\n\n"
        f"질문: {query}\n\n"
        f"답변:"
    )
    
    # Ollama LLM 호출 (모델 이름은 환경에 맞게 조정)
    llm = Ollama(model="llama2")
    answer = llm(prompt)
    return answer

# RAG용 벡터스토어 생성 (전체 CSV 데이터를 대상으로)
vectorstore = build_vectorstore(df)

st.markdown("---")
st.header("🔎 RAG 질의 응답")
rag_query = st.text_input("RAG 질문을 입력하세요", key="rag_query")
if rag_query:
    response = answer_query(rag_query, vectorstore)
    st.markdown("### 답변:")
    st.write(response)
