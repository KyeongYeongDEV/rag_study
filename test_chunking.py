from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

long_text = """
[1. 사과에 대하여]
사과는 장미과에 속하는 과일로, 전 세계적으로 가장 널리 섭취되는 과일 중 하나입니다.
사과의 원산지는 중앙아시아로 알려져 있으며, 수천 년 동안 재배되어 왔습니다.
특히 '아침 사과는 금사과'라는 말이 있을 정도로 식이섬유와 비타민 C가 풍부하여 건강에 매우 유익합니다.
하지만 사과 씨앗에는 미량의 독성이 있으므로 섭취 시 주의해야 합니다.

[2. 바나나의 특징]
바나나는 파초과에 속하는 열대 과일입니다. 길쭉하고 구부러진 모양이 특징이며, 익으면 노란색을 띠게 됩니다.
바나나는 칼륨이 풍부하여 나트륨 배출을 돕고 혈압 조절에 도움을 줍니다.
또한 휴대가 간편하고 소화가 잘 되어 운동 선수들의 에너지 보충용으로 인기가 많습니다.
껍질에 검은 반점(슈가 스팟)이 생겼을 때 당도가 가장 높고 면역력 강화 성분도 많아집니다.

[3. RAG 시스템 구축 시 주의사항]
RAG(검색 증강 생성) 시스템을 구축할 때 가장 중요한 것은 데이터의 전처리입니다.
데이터를 너무 작게 자르면 문맥(Context)을 잃어버리고, 너무 크게 자르면 불필요한 노이즈가 섞입니다.
따라서 적절한 청킹(Chunking) 전략을 사용하는 것이 검색 품질을 좌우합니다.
특히 Overlap(중첩)을 설정하여 잘리는 부분의 문맥이 이어지도록 하는 것이 필수적입니다.
"""

print("[Case 1] 단순 문자 단위 자르기 (Fixed Size)")
print("   → 문맥 무시하고 글자 수(100자) 채우면 무조건 자름")

splitter_v1 = CharacterTextSplitter(
    separator="",  # 구분자 없음 - 글자 수대로 자름
    chunk_size=100,
    chunk_overlap=0
)
docs_v1 = splitter_v1.create_documents([long_text])

for i, doc in enumerate(docs_v1[:3]):
    print(f"✂️ Chunk {i+1}:")
    print(f"'{doc.page_content}'")
    print("-" * 30)

print("\n" + "="*50 + "\n")

# 똑똑하게 자르기 (RecursiveCharacterTextSplitter)
print("[Case 2] 재귀적 자르기 (Recursive)")
print("   → 문단 > 문장 > 단어 순으로 최대한 의미를 유지하며 자름")

# chunk_size=100: 목표는 100글자
# chunk_overlap=20: 잘리는 부분 앞뒤로 20글자는 겹치게 
splitter_v2 = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20, 
    separators=["\n\n", "\n", " ", ""] # 우선순위: 문단 -> 문장 -> 공백
)
docs_v2 = splitter_v2.create_documents([long_text])

for i, doc in enumerate(docs_v2[:3]):
    print(f"Chunk {i+1}:")
    print(f"'{doc.page_content}'")
    print("-" * 30)