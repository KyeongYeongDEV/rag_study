from sentence_transformers import CrossEncoder


#  리랭커 모델 로드
# RAG에서 가장 널리 쓰이는 가성비 모델 (MS MARCO 데이터셋으로 학습됨)
# 이 모델은 (질문, 문서)를 쌍으로 묶어서 정답 확률을 계산합니다.
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


# 상황 설정: 검색 엔진이 대충 가져온 결과
query = "서울의 인구는 얼마인가요?"

# 상황: 검색 엔진이 '인구', '서울' 단어만 보고 관련 없는 문서들을 상위에 노출시킴
# 실제 정답(940만 명)은 3등에 섞여 있다고 가정합니다.
documents = [
    "부산의 인구는 약 330만 명입니다.",        
    "경기도의 인구는 1,360만 명으로 가장 많습니다.", 
    "서울의 면적은 605.2km² 입니다.", 
    "서울은 대한민국의 수도이며 한강이 흐릅니다.", 
    "2023년 기준 서울특별시의 인구는 약 940만 명으로 집계되었습니다.",         
]

print(f"질문: '{query}'")
print("[Before] 검색 엔진이 가져온 원래 순서:")
for i, doc in enumerate(documents):
    print(f"   {i+1}등: {doc}")

print("-" * 50)

# 리랭킹 (Reranking) 수행
# (질문, 문서)를 쌍으로 묶어서 입력
inputs = [[query, doc] for doc in documents]

# 모델이 채점한 점수
scores = model.predict(inputs)

# 점수가 높은 순서대로 다시 정렬 
ranked_results = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)

print("[After] 리랭커(Cross-Encoder)가 고친 순서:")
for i, (score, doc) in enumerate(ranked_results):
    print(f"   {i+1}등 (점수: {score:.4f}): {doc}")