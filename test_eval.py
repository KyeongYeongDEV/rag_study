import google.generativeai as genai
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


GOOGLE_API_KEY = "발급 받은 key 입력"
genai.configure(api_key=GOOGLE_API_KEY)

es = Elasticsearch("http://localhost:9200")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
index_name = "fruit_vector_shop"

# 학생 RAG 로직 (MMR + 답변 생성)
# 기존 MMR 함수 재사용
def mmr_sort(docs, query_vector, lambda_param=0.5, top_k=3):
    if len(docs) == 0: return []
    selected_indices = []
    candidate_indices = list(range(len(docs)))
    doc_vectors = np.array([d['_source']['description_vector'] for d in docs])

    sims_to_query = cosine_similarity([query_vector], doc_vectors)[0]
    best_doc_idx = np.argmax(sims_to_query)
    selected_indices.append(best_doc_idx)
    candidate_indices.remove(best_doc_idx)

    for _ in range(top_k - 1):
        if not candidate_indices: break
        mmr_scores = []
        for cand_idx in candidate_indices:
            relevance = sims_to_query[cand_idx]
            sims_to_selected = cosine_similarity([doc_vectors[cand_idx]], doc_vectors[selected_indices])[0]
            redundancy = np.max(sims_to_selected)
            score = (lambda_param * relevance) - ((1 - lambda_param) * redundancy)
            mmr_scores.append(score)
        best_candidate_idx = candidate_indices[np.argmax(mmr_scores)]
        selected_indices.append(best_candidate_idx)
        candidate_indices.remove(best_candidate_idx)
    return [docs[i] for i in selected_indices]

def run_rag(question):
    query_vec = model.encode(question)
    response = es.search(index=index_name, knn={
        "field": "description_vector", "query_vector": query_vec,
        "k": 10, "num_candidates": 100
    })

    final_docs = mmr_sort(response['hits']['hits'], query_vec, lambda_param=0.5, top_k=3)

    context_text = ""
    for i, doc in enumerate(final_docs):
        context_text += f"{i+1}. {doc['_source']['name']}: {doc['_source']['description']}"

    prompt = f"""
    [참고 정보]
    {context_text}
    질문: {question}
    위 정보를 바탕으로 핵심만 간결하게 답변해 주세요.
    """
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    return gemini_model.generate_content(prompt).text

# 선생님 AI: 심판(Judge) 구현
def evaluate_answer(question, ground_truth, ai_answer):
    """
    Gemini에게 '채점관' 페르소나를 부여하여 점수와 이유를 출력하게 함
    """
    judge_prompt = f"""
    당신은 공정한 채점관입니다.
    아래의 [질문]에 대해 [AI 답변]이 [모범 답안]과 얼마나 일치하는지 평가하세요.

    [질문]: {question}
    [모범 답안]: {ground_truth}
    [AI 답변]: {ai_answer}

    평가 기준:
    1. 정확성: 모범 답안의 핵심 내용이 포함되었는가?
    2. 환각(Hallucination): 없는 내용을 지어내지 않았는가?

    반드시 다음 형식으로만 출력하세요:
    점수: (1~10점) / 이유: (한 줄 평가)
    """
    judge_model = genai.GenerativeModel('gemini-2.5-flash')
    return judge_model.generate_content(judge_prompt).text.strip()

# 테스트 데이터셋 (Ground Truth)
test_set = [
    {
        "question": "사과와 바나나의 색깔 차이는?",
        "ground_truth": "사과는 빨간색이고, 바나나는 노란색입니다."
    },
    {
        "question": "아이폰은 과일인가요?",
        "ground_truth": "아니요, 아이폰은 과일이 아니라 스마트폰 전자기기입니다."
    },
    {
        "question": "아침에 먹으면 좋은 과일은?",
        "ground_truth": "사과입니다. 아침에 먹으면 건강에 좋다고 알려져 있습니다."
    }
]

# 평가 실행
print("RAG 성능 평가(LLM-as-a-Judge)를 시작합니다")
total_score = 0

for i, test in enumerate(test_set):
    print(f"문제 {i+1}: {test['question']}")

    # 1. RAG가 문제 풀기 (Student)
    my_rag_answer = run_rag(test['question'])
    print(f"   학생(RAG) 답안: {my_rag_answer.strip()}")

    # 2. Judge가 채점하기 (Teacher)
    eval_result = evaluate_answer(test['question'], test['ground_truth'], my_rag_answer)
    print(f"   선생님(Judge) 평가: {eval_result}")

    # 점수 파싱 및 합산
    if "점수:" in eval_result:
        try:
            score_part = eval_result.split("점수:")[1].split("/")[0]
            score = int(score_part.replace("점", "").strip())
            total_score += score
        except: pass

print("="*50)
print(f"최종 성적표: 평균 {total_score / len(test_set):.1f}점 / 10점 만점")