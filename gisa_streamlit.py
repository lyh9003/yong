import streamlit as st
import pandas as pd
import datetime
import time
import openai
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter

# OpenAI API 키 설정 (환경변수 또는 직접 입력 가능)
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
    """FAISS 벡터 저장소를 생성합니다."""
    df = df.dropna(subset=['title', 'summary'])
    df['content'] = df['title'] + "\n" + df['summary']
    
    # LangChain 데이터 로더 적용
    loader = DataFrameLoader(df, page_content_column='content')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    
    # FAISS 벡터 저장소 구축
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_documents(split_docs, embeddings)
    return vector_db

def query_rag(vector_db, query):
    """RAG를 사용하여 뉴스 기사 요약을 생성합니다."""
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    llm = OpenAI(temperature=0.3)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    response = qa_chain.run(query)
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
