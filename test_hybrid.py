from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# 연결 및 모델 로드
es = Elasticsearch("http://localhost:9200")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
index_name = "fruit_vector_shop" 

# RRF 알고리즘 함수 (등수 기반 점수 병합)
def rrf_merge(bm25_hits, vector_hits, k=60, weight_bm25=1.0, weight_vector=1.0):
    scores = {} # 문서 ID별 최종 점수를 담을 딕셔너리

    # BM25 결과 처리 (가중치 반영)
    for rank, hit in enumerate(bm25_hits):
        doc_id = hit['_id']
        # 등수가 높을수록 점수가 커지는 공식
        score = weight_bm25 * (1 / (k + rank + 1))
        scores[doc_id] = scores.get(doc_id, 0) + score

    # Vector 결과 처리 (가중치 반영)
    for rank, hit in enumerate(vector_hits):
        doc_id = hit['_id']
        score = weight_vector * (1 / (k + rank + 1))
        scores[doc_id] = scores.get(doc_id, 0) + score

    # 점수 높은 순으로 정렬
    sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_docs

# 하이브리드 검색 실행
keyword = "사과"
print(f" '{keyword}' 하이브리드 검색 시작")

# BM25로 찾기 
response_bm25 = es.search(
    index=index_name,
    query={"match": {"description": keyword}},
    size=3
)
bm25_hits = response_bm25['hits']['hits']
print("[BM25 결과]")
for h in bm25_hits:
    print(f" - {h['_source']['name']}")

# Vector로 찾기
query_vector = model.encode(keyword)
knn_query = {
    "field": "description_vector",
    "query_vector": query_vector,
    "k": 3,
    "num_candidates": 100
}
response_vector = es.search(index=index_name, knn=knn_query)
vector_hits = response_vector['hits']['hits']
print("[Vector 결과]")
for h in vector_hits:
    print(f" - {h['_source']['name']}")


# RRF로 합치기
print("두 결과를 RRF 알고리즘으로 합치는 중")
final_results = rrf_merge(bm25_hits, vector_hits)

print(f"[최종 하이브리드 랭킹] (Top 3):")
for rank, (doc_id, score) in enumerate(final_results[:3]):
    # ID로 상세 정보 다시 가져오기
    doc = es.get(index=index_name, id=doc_id)['_source']
    print(f"{rank+1}등 [RRF점수: {score:.5f}] {doc['name']} - {doc['description']}")