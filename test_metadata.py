from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

es = Elasticsearch("http://localhost:9200")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
index_name = "metadata_shop"

# 인덱스 생성 (Mapping 정의)
# 필터링을 하려면 데이터 타입(keyword, integer)을 명확히 알려줘야 합니다.
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)

mapping = {
    "mappings": {
        "properties": {
            "name": {"type": "text"},
            "description": {"type": "text"},
            "description_vector": {"type": "dense_vector", "dims": 384},
            # 필터링을 위한 필드 정의
            "category": {"type": "keyword"},  # 정확히 일치해야 하는 태그
            "price": {"type": "integer"}      # 숫자 범위 검색용
        }
    }
}
es.indices.create(index=index_name, body=mapping)

# 데이터 넣기 
data = [
    {
        "name": "청송 꿀 사과",
        "description": "아침에 먹으면 좋은 달콤한 빨간 사과입니다.",
        "category": "fruit",  # 과일
        "price": 2000
    },
    {
        "name": "아이폰 15",
        "description": "애플이 만든 최신 스마트폰, 혁신적인 디자인.",
        "category": "electronics", # 전자기기
        "price": 1200000
    },
    {
        "name": "유기농 바나나",
        "description": "달콤하고 부드러운 노란 바나나.",
        "category": "fruit",  # 과일
        "price": 3000
    },
    {
        "name": "맥북 프로",
        "description": "전문가를 위한 강력한 성능의 애플 노트북.",
        "category": "electronics", # 전자기기
        "price": 3500000
    }
]

for i, item in enumerate(data):
    item["description_vector"] = model.encode(item["description"])
    es.index(index=index_name, id=i, document=item)

es.indices.refresh(index=index_name)
print("✅ 데이터 추가 완료!")

# 필터 X vs 필터 O)

query_text = "애플 제품 추천해줘"
query_vec = model.encode(query_text)

print(" User Question: '{query_text}'")

# [Case 1] 필터 없이 검색
print("[Case 1] 필터 없음 (Vector Only)")
res_no_filter = es.search(index=index_name, knn={
    "field": "description_vector",
    "query_vector": query_vec,
    "k": 3,
    "num_candidates": 100
})

for hit in res_no_filter['hits']['hits']:
    print(f"   - {hit['_source']['name']} ({hit['_source']['category']})")

print("-" * 50)

# [Case 2] 필터 적용
target_category = "fruit"
print("[Case 2] 메타데이터 필터링 (Category='{target_category}')")
print("   → '애플'이라고 검색했지만, 과일 중에서만 찾습니다.")

res_filter = es.search(index=index_name, knn={
    "field": "description_vector",
    "query_vector": query_vec,
    "k": 3,
    "num_candidates": 100,
    #  SQL의 WHERE 절과 같은 역할
    "filter": { 
        "term": {"category": target_category}
    }
})

for hit in res_filter['hits']['hits']:
    print(f"   - {hit['_source']['name']} ({hit['_source']['category']})")