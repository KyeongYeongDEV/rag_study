import time
import numpy as np
import faiss

# 가짜 데이터 생성 
d = 128      # 벡터의 차원 
nb = 100000  # 데이터 개수 (10만 개)
nq = 1       # 질문 개수 (1개)

# 랜덤 벡터 데이터 생성
database_vectors = np.random.random((nb, d)).astype('float32')
query_vector = np.random.random((nq, d)).astype('float32')

# 느린 검색 - Brute Force
print("[Case 1] 전수 조사")

# 모든 거리 계산
index_flat = faiss.IndexFlatL2(d) 
index_flat.add(database_vectors) 

start_time = time.time()
D, I = index_flat.search(query_vector, k=5) # 검색 수행
end_time = time.time()

print(f"   소요 시간: {end_time - start_time:.6f}초")
print(f"   찾은 인덱스: {I[0]}")

# 빠른 검색 - HNSW
print("[Case 2] 고속 검색")

# HNSW 인덱스 생성 (M=32: 연결 다리 개수)
M = 32 
index_hnsw = faiss.IndexHNSWFlat(d, M)

# HNSW는 데이터를 쌓을 때 시간이 조금 걸린다 (지하철 노선 까는 작업 필요)
# 하지만 한 번 깔아두면 검색은 엄청 빠름
index_hnsw.hnsw.efConstruction = 256 # 정밀도
index_hnsw.add(database_vectors) 

start_time = time.time()
D_hnsw, I_hnsw = index_hnsw.search(query_vector, k=5)
end_time = time.time()

print(f"   소요 시간: {end_time - start_time:.6f}초")
print(f"   찾은 인덱스: {I_hnsw[0]}")