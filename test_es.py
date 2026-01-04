from elasticsearch import Elasticsearch

# Elasticsearch 연결
es = Elasticsearch("http://localhost:9200")

# 인덱스(저장소) 생성
index_name = "fruit_shop"

if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)

es.indices.create(index=index_name)
print(f"'{index_name}' 인덱스 생성 완료")


# 데이터 넣기
data = [
    {"name": "맛있는 사과", "description": "아침에 먹으면 건강에 좋은 빨간 사과"},
    {"name": "달콤한 바나나", "description": "노랗고 달콤한 열대 과일"},
    {"name": "시원한 수박", "description": "여름철 갈증 해소에 최고"},
    {"name": "아이폰", "description": "사과 로고가 있는 스마트폰 전자기기"}
]

for i, item in enumerate(data):
    es.index(index=index_name, id=i, document=item)
print("데이터 적재 완료")

# 강제 새로고침(Refresh) 
# 이 코드가 없으면 방금 넣은 데이터가 검색 안 될 수 있음
es.indices.refresh(index=index_name)
print("인덱스 새로고침 완료")
# ==========================================

# 검색하기 (BM25 방식)
keyword = "사과"
query = {
    "query": {
        "match": {
            "description": keyword
        }
    }
}

print(f"\n '{keyword}' 검색 결과:") 
response = es.search(index=index_name, body=query)

for hit in response['hits']['hits']:
    score = hit['_score']
    doc = hit['_source']
    print(f"[점수: {score:.2f}] {doc['name']} - {doc['description']}")