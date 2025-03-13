import streamlit as st
import pandas as pd
import datetime
import time
import openai
import pysqlite3
import sys
import os
sys.modules['sqlite3'] = pysqlite3  # sqlite3 모듈을 최신 버전으로 교체

import sqlite3  # 이제 최신 sqlite3 사용 가능
# LangChain 관련 라이브러리
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

# OpenAI API 키 설정 (Streamlit secrets에 등록)
openai.api_key = st.secrets["OPENAI_API_KEY"]

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

# ===============================================
# OpenAI를 활용한 헬퍼 함수들
# ===============================================
def check_semiconductor(question):
    """
    질문이 반도체와 관련된지 확인합니다.
    OpenAI API에 '예' 또는 '아니오'로 대답하도록 요청합니다.
    """
    prompt = f"다음 질문이 반도체와 관련이 있으면 '예', 아니면 '아니오'로 대답해줘:\n{question}"
    response = openai.Completion.create(
        model="GPT-4o-mini",
        prompt=prompt,
        max_tokens=3,
        temperature=0
    )
    answer = response.choices[0].text.strip()
    return answer == "예"

def extract_keyword(question):
    """
    질문에서 핵심 키워드를 한 단어로 추출합니다.
    """
    prompt = f"다음 질문에서 핵심 키워드를 한 단어로 추출해줘:\n{question}"
    response = openai.Completion.create(
        model="GPT-4o-mini",
        prompt=prompt,
        max_tokens=5,
        temperature=0
    )
    keyword = response.choices[0].text.strip()
    return keyword

def generate_answer_openai(question):
    """
    일반적인 질문에 대해 OpenAI를 활용하여 답변 생성합니다.
    """
    response = openai.Completion.create(
        model="GPT-4o-mini",
        prompt=question,
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].text.strip()

def generate_answer_with_rag(question, context):
    """
    뉴스 기사 내용을 context로 하여 RAG 방식의 답변을 생성합니다.
    """
    prompt = f"주어진 뉴스 기사 내용을 참고하여 아래 질문에 대해 답변을 생성해줘.\n\n뉴스 기사:\n{context}\n\n질문:\n{question}\n\n답변:"
    response = openai.Completion.create(
        model="GPT-4o-mini",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# ===============================================
# 뉴스 기사 데이터셋 벡터 스토어 생성 (Chroma)
# ===============================================


CHROMA_PERSIST_DIR = "./chroma_db"  # 상대 경로 사용

# 디렉터리 생성
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)



def build_vector_store(df):
    """
    각 기사의 제목과 요약을 하나의 문서로 결합한 후,
    LangChain의 OpenAIEmbeddings를 이용해 Chroma 벡터 스토어를 구축합니다.
    """
    documents = []
    for idx, row in df.iterrows():
        content = f"제목: {row['title']}\n요약: {row.get('summary', '요약 정보가 없습니다.')}"
        documents.append(Document(page_content=content, metadata={"키워드": row["키워드_목록"], "date": row["date"]}))
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        documents,
        embeddings,
        collection_name="news_articles",
        persist_directory=CHROMA_PERSIST_DIR  # 추가 설정
    )
    return vector_store

# 데이터 불러오기
df = load_data()

# 최근 1주일치 기사 필터링용 데이터 준비
if not df.empty:
    max_date = df['date'].max()
    one_week_ago = max_date - datetime.timedelta(days=7)
    default_recent_df = df[df['date'] >= one_week_ago]
else:
    default_recent_df = df.copy()

# 벡터 스토어 생성 (RAG 검색에 활용)
if not df.empty:
    vector_store = build_vector_store(df)

# ======================================================
# Streamlit 앱 타이틀 및 기본 설정
# ======================================================
st.title("📢 반도체 뉴스 업데이트")
st.write("yh9003.lee@samsung.com")

# ======================================================
# [추가] 질문 입력란 및 RAG/일반 검색 분기 처리
# ======================================================
st.header("❓ 질문 입력")
question = st.text_input("질문을 입력하세요:")

if question:
    if check_semiconductor(question):
        st.write("**반도체 관련 질문으로 인식하였습니다.**")
        # 질문에서 핵심 키워드 추출
        keyword = extract_keyword(question)
        st.write(f"추출된 키워드: **{keyword}**")
        # 벡터 스토어를 활용하여 해당 키워드와 가장 밀접한 기사 검색
        docs = vector_store.similarity_search(keyword, k=1)
        if docs:
            context = docs[0].page_content
            st.write("**검색된 뉴스 기사:**")
            st.write(context)
            # RAG 방식으로 답변 생성
            answer = generate_answer_with_rag(question, context)
            st.write("**RAG를 통한 답변:**")
            st.write(answer)
        else:
            st.write("해당 키워드와 관련된 뉴스 기사가 없습니다.")
    else:
        st.write("**반도체 관련 질문이 아니라고 판단되어 일반 OpenAI 검색을 진행합니다.**")
        answer = generate_answer_openai(question)
        st.write("**답변:**")
        st.write(answer)

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
# 3) 필터 적용 (날짜 + 키워드)
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

st.write(f"**총 기사 수:** {len(filtered_df)}개")

# ======================================================
# 4) 날짜별 → 키워드별 → 기사 목록 표시
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
