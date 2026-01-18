import os
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

GOOGLE_API_KEY = "발급 받은 key 입력"

# 설정 및 연결
if GOOGLE_API_KEY.startswith("AIza"):
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("⚠️ 경고: 올바른 Google API 키가 설정되지 않았습니다.")

es = Elasticsearch("http://localhost:9200")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
index_name = "fruit_vector_shop"

# RRF 함수 
def rrf_merge(bm25_hits, vector_hits, k=60, weight_bm25=1.0, weight_vector=1.0):
    scores = {}
    for rank, hit in enumerate(bm25_hits):
        doc_id = hit['_id']
        scores[doc_id] = scores.get(doc_id, 0) + weight_bm25 * (1 / (k + rank + 1))
    for rank, hit in enumerate(vector_hits):
        doc_id = hit['_id']
        scores[doc_id] = scores.get(doc_id, 0) + weight_vector * (1 / (k + rank + 1))
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)

# 검색 수행 (Retrieval)
question = "사과랑 바나나의 차이점을 설명해줘."
print(f"User: {question}\n")

print(" 1. 관련된 문서를 검색 중입니다...")

# BM25 & Vector 검색
res_bm25 = es.search(index=index_name, query={"match": {"description": question}}, size=3)
res_vec = es.search(index=index_name, knn={
    "field": "description_vector", "query_vector": model.encode(question),
    "k": 3, "num_candidates": 100
})
# RRF 합치기
final_docs = rrf_merge(res_bm25['hits']['hits'], res_vec['hits']['hits'])

#  프롬프트 구성 (CoT 적용)
context_text = ""
print("\n📄 [참고할 문서 목록]")
for i, (doc_id, score) in enumerate(final_docs[:3]):
    doc = es.get(index=index_name, id=doc_id)['_source']
    print(f" - {doc['name']}")
    context_text += f"{i+1}. {doc['name']}: {doc['description']}\n"

# Gemini에게 보낼 프롬프트 
# 여기에서 단계별로 생각하라고 알려주는 게 Cot 방식입니다.
prompt = f"""
당신은 논리적인 과일 가게 점원입니다.
아래 [참고 정보]를 바탕으로 사용자의 질문에 답변하세요.

반드시 다음 단계(Step-by-Step)에 따라 생각하고 답변을 작성하세요:
1. [참고 정보]에서 질문과 관련된 핵심 키워드를 찾으세요.
2. 각 과일의 특징을 비교 분석하세요.
3. 최종적으로 차이점을 요약하여 친절하게 설명하세요.

[참고 정보]
{context_text}

[사용자 질문]
{question}
"""

print("\n📨 2. AI에게 보낼 프롬프트가 준비되었습니다!")
print("-" * 50)
print(prompt)
print("-" * 50)

# Gemini에게 답변 요청 (Generation)
if not GOOGLE_API_KEY.startswith("AIza"):
    print("\n⚠️ API 키가 없어서 실제 답변 생성을 건너뜁니다.")
else:
    print("\n🤖 3. Gemini가 생각하는 중입니다... (CoT 작동 중)")
    
    # 모델 선택 (gemini-1.5-flash가 빠르고 무료 사용량이 넉넉합니다)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    
    try:
        response = gemini_model.generate_content(prompt)
        
        print("\n✅ [Gemini의 최종 답변]")
        print("=" * 50)
        print(response.text)
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")