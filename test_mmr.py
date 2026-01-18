import numpy as np
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

#설정 및 모델 로드
es = Elasticsearch("http://localhost:9200")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
index_name = "fruit_vector_shop"

# 데이터 추가
new_data = [
    {"name": "청송 꿀 사과", "description": "꿀이 가득 들어있는 맛있는 빨간 사과"},
    {"name": "아침의 사과", "description": "매일 아침 건강을 챙겨주는 붉은 과일"},
    {"name": "풋사과", "description": "아직 덜 익었지만 싱그러운 초록 사과"},
    {"name": "홍옥 사과", "description": "새콤달콤한 맛이 일품인 빨간 과일"}
]

for i, item in enumerate(new_data):
    item["description_vector"] = model.encode(item["description"])
    es.index(index=index_name, id=100+i, document=item)

es.indices.refresh(index=index_name)
print("✅ 데이터 추가 완료\n")


#MMR 알고리즘 함수
def mmr_sort(docs, query_vector, lambda_param=0.5, top_k=3):
    if len(docs) == 0: return []

    selected_indices = []
    candidate_indices = list(range(len(docs)))

    doc_vectors = np.array([d['_source']['description_vector'] for d in docs])
    
    # 첫 번째는 가장 정확한 문서를 뽑는다.
    sims_to_query = cosine_similarity([query_vector], doc_vectors)[0]
    best_doc_idx = np.argmax(sims_to_query)
    
    selected_indices.append(best_doc_idx)
    candidate_indices.remove(best_doc_idx)

    # 두 번째부터는 MMR 공식 적용
    for _ in range(top_k - 1):
        if not candidate_indices: break
            
        mmr_scores = []
        for cand_idx in candidate_indices:
            # 정확도 - Query와 얼마나 비슷한지
            relevance = sims_to_query[cand_idx]
            
            # 중복도 - 이미 뽑힌 애들과 얼마나 겹치는지
            # 이미 선택된 문서들과의 유사도 중 가장 큰 값을 찾는다.
            sims_to_selected = cosine_similarity(
                [doc_vectors[cand_idx]], 
                doc_vectors[selected_indices]
            )[0]
            redundancy = np.max(sims_to_selected)
            
            # MMR 점수 계산
            score = (lambda_param * relevance) - ((1 - lambda_param) * redundancy)
            mmr_scores.append(score)
            
        # MMR 점수가 가장 높은 문서 선택
        best_candidate_idx = candidate_indices[np.argmax(mmr_scores)]
        selected_indices.append(best_candidate_idx)
        candidate_indices.remove(best_candidate_idx)

    return [docs[i] for i in selected_indices]

# 일반 검색 vs MMR 검색 비교
query = "사과"
query_vec = model.encode(query)

print(f"User Question: '{query}'\n")

# 후보 10개 가져옴
response = es.search(index=index_name, knn={
    "field": "description_vector", "query_vector": query_vec,
    "k": 10, "num_candidates": 100
})
candidates = response['hits']['hits']

# 일반 검색
print(f"[일반 Vector 검색] (Top 3)")
for hit in candidates[:3]:
    print(f"   - {hit['_source']['name']}")

print("-" * 50)

# MMR 검색
print(f"[MMR 검색] (Top 3, Lambda=0.5)")
mmr_results = mmr_sort(candidates, query_vec, lambda_param=0.5, top_k=3)

for hit in mmr_results:
    print(f"   - {hit['_source']['name']}")