import os
import time  
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


genai.configure(api_key=GOOGLE_API_KEY)

def rewrite_query(history, current_query):
    prompt = f"""
    당신은 검색 쿼리 최적화 전문가입니다.
    사용자와의 [이전 대화]를 참고하여, [현재 질문]이 무엇을 의미하는지 명확하게 파악하세요.
    그리고 검색 엔진이 이해하기 쉽도록 '완전한 문장'으로 다시 작성해 주세요.
    
    [이전 대화]
    {history}
    
    [현재 질문]
    {current_query}
    
    [규칙]
    1. 대명사(그거, 이건, 걔는)를 구체적인 명사로 바꾸세요.
    2. 답변은 오직 '수정된 검색어'만 출력하세요. (설명 금지)
    """

    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    return response.text.strip()

# 채팅 시뮬레이션 

chat_history = [] 

# 시나리오 1
q1 = "아이폰 15 프로의 특징이 뭐야?"
print(f"👤 사용자: {q1}")
rewritten_q1 = rewrite_query(chat_history, q1)
print(f"🤖 최적화된 검색어: {rewritten_q1}\n")
chat_history.append(f"User: {q1}")

# API 제한 방지
time.sleep(3)


# 시나리오 2
q2 = "그거 가격은 얼마야?"
print(f"👤 사용자: {q2}")
rewritten_q2 = rewrite_query(chat_history, q2)
print(f"🤖 최적화된 검색어: {rewritten_q2}\n")
chat_history.append(f"User: {q2}")

time.sleep(3)


# 시나리오 3 심화 - 과제 적용
# 과제: "그거랑 갤럭시랑 비교해줘" 로 변경
q3 = "그거랑 갤럭시랑 비교해줘" 
print(f"👤 사용자: {q3}")
rewritten_q3 = rewrite_query(chat_history, q3)
print(f"🤖 최적화된 검색어: {rewritten_q3}")