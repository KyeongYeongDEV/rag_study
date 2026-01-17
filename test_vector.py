from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# AI 모델 준비 (텍스트 -> 384개의 숫자로 바꿔주는 기계)
print("AI 모델을 로딩하고 있습니다. 잠시만 기다려주세요!")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
print("모델 로딩 완료!")

# Elasticsearch 연결
es = Elasticsearch("http://localhost:9200")

# 벡터 전용 저장소(인덱스) 만들기
index_name = "fruit_vector_shop"

# 기존에 있으면 삭제하고 새로 만든다.
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)

# 저장소 지도(Mapping) 그리기
# "description_vector"라는 칸에는 384개의 숫자가 들어간다고 미리 알려준다.
mappings = {
    "properties": {
        "name": {"type": "text"},
        "description": {"type": "text"},
        "description_vector": {
            "type": "dense_vector", # 벡터(숫자)를 담을 타입
            "dims": 384,            # 이 모델은 숫자를 384개 뱉는다
            "index": True,          # 검색 가능하게 설정
            "similarity": "cosine"  # 코사인 유사도로 비교
        }
    }
}

es.indices.create(index=index_name, mappings=mappings)
print(f"'{index_name}' 벡터 저장소 생성 완료")

# 데이터 준비
data = [
    {"name": "맛있는 사과", "description": "아침에 먹으면 건강에 좋은 빨간 사과"},
    {"name": "달콤한 바나나", "description": "노랗고 달콤한 열대 과일"},
    {"name": "시원한 수박", "description": "여름철 갈증 해소에 최고"},
    {"name": "아이폰", "description": "사과 로고가 있는 스마트폰 전자기기"}
]

# 임베딩 - 데이터를 벡터로 변환해서 저장하기 
print("데이터를 벡터로 변환해서 저장 중")
for i, item in enumerate(data):
    # 문장(description)을 AI 모델에 넣어서 숫자(vector)로 바꾼다.
    vector = model.encode(item["description"])
    
    # 문서에 벡터도 같이 담아서 저장
    item["description_vector"] = vector
    es.index(index=index_name, id=i, document=item)

# 검색 가능하게 새로고침
es.indices.refresh(index=index_name)
print("데이터 적재 완료!")

# 의미로 검색하기 (Vector Search)
keyword = "사과"
print(f"\n '{keyword}' 검색 (벡터 방식):")

# 검색어도 숫자로 변환해야 비교할 수 있습니다.
query_vector = model.encode(keyword)

# k-NN 검색 (나랑 가장 가까운 이웃 3개 찾기)
knn_query = {
    "field": "description_vector", # 벡터가 저장된 곳
    "query_vector": query_vector,  # 검색어의 벡터
    "k": 3,                        # 상위 3개만 가져와
    "num_candidates": 100
}

response = es.search(index=index_name, knn=knn_query)

for hit in response['hits']['hits']:
    score = hit['_score']
    doc = hit['_source']
    # 벡터 값은 너무 기니까(숫자 384개) 출력하지 않고 이름과 설명만 봅니다.
    print(f"[유사도: {score:.4f}] {doc['name']} - {doc['description']}")