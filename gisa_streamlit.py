import streamlit as st
import pandas as pd
import datetime
import time
import openai  # 필요시 다른 API 설정에도 활용 가능

# 새 라이브러리 임포트
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# 기존 텍스트 분할 도구 (필요시 계속 사용)
from langchain.text_splitter import CharacterTextSplitter

# OpenAI API 키 설정 (다른 API 키가 필요한 경우 변경)
openai.api_key = st.secrets["OPENAI_API_KEY"]

GITHUB_CSV_URL = f"https://raw.githubusercontent.com/lyh9003/yong/main/Total_Filtered_No_Comment.csv?nocache={int(time.time())}"

def load_data():
    """CSV를 불러와 DataFrame으로 반환합니다."""
    df = pd.read_csv(GITHUB_CSV_URL, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values(by='date', ascending=False)
    df['키워드_목록'] = df['키워드'].apply(lambda x: [k.strip() for k in str(x).split(',')] if pd.notna(x) else [])
    return df

def create_vector_db(df):
    """Chroma 벡터 저장소를 생성합니다."""
    df = df.dropna(subset=['title', 'summary'])
    df['content'] = df['title'] + "\n" + df['summary']
    
    # DataFrame의 각 행을 Document 객체로 변환
    documents = [
        Document(page_content=row['content'], metadata=row.to_dict())
        for idx, row in df.iterrows()
    ]
    
    # 텍스트 분할 (chunking)
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    
    # OllamaEmbeddings를 사용하여 임베딩 생성 후, Chroma 벡터 저장소 구축
    embeddings = OllamaEmbeddings()
    vector_db = Chroma.from_documents(split_docs, embeddings)
    return vector_db

def query_rag(vector_db, query):
    """새 라이브러리를 사용하여 RAG 기반 뉴스 기사 요약을 생성합니다."""
    # 1. 벡터 저장소에서 관련 문서 검색 (예: 상위 5개)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.get_relevant_documents(query)
    
    # 2. CrossEncoder를 이용해 문서 재정렬 (HuggingFace 모델 사용)
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = CrossEncoderReranker(cross_encoder=cross_encoder)
    reranked_docs = reranker.rerank(query, retrieved_docs)
    
    # 3. 재정렬된 문서들의 내용을 하나의 컨텍스트로 결합
    context = "\n\n".join([doc.page_content for doc in reranked_docs])
    
    # 4. ChatPromptTemplate을 사용하여 프롬프트 구성
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that summarizes news articles based on provided context."),
        ("user", "Given the following context:\n{context}\n\nPlease summarize the key points answering: {query}")
    ])
    prompt = prompt_template.format(context=context, query=query)
    
    # 5. ChatOllama를 사용하여 답변 생성 (예시로 'llama2' 모델 사용)
    llm = ChatOllama(model="llama2", temperature=0.3)
    response = llm(prompt)
    return response

# Streamlit UI 설정
df = load_data()
vector_db = create_vector_db(df)

st.title("📢 반도체 뉴스 RAG 기반 검색")
st.write("문의/아이디어 : yh9003.lee@samsung.com")

# 사용자 입력
query = st.text_input("🔍 검색할 내용을 입력하세요:")
if query:
    with st.spinner("🔎 검색 중..."):
        response = query_rag(vector_db, query)
    st.markdown("### 📜 검색 결과 요약")
    st.write(response)

# 기본적인 뉴스 리스트 출력
st.write(f"**총 기사 수:** {len(df)}개")
grouped_by_date = df.groupby(df['date'].dt.date, sort=False)
for current_date, date_group in grouped_by_date:
    st.markdown(f"## {current_date.strftime('%Y-%m-%d')}")
    for idx, row in date_group.iterrows():
        with st.expander(f"📰 {row['title']}"):
            st.write(f"**요약:** {row.get('summary', '요약 정보가 없습니다.')}")
            link = row.get('link', None)
            if pd.notna(link):
                st.markdown(f"[🔗 기사 링크]({link})")
