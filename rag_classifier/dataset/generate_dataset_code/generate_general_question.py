import openai  # 또는 해당 LLM 클라이언트 라이브러리
import json
import os
from openai import OpenAI

# 환경변수로 API 키 설정되어 있다고 가정
client = OpenAI(api_key = '')

def generate_questions(option):
    if option == "English":
        prompt = """
        You are a helpful assistant.

        Generate **100 distinct short questions** that do **not require retrieving any external documents or specific reports.**

        The questions should cover a wide variety of simple, general topics that can be answered from common sense or general knowledge, such as:
        - Everyday small talk (e.g., greetings, hobbies, daily life)
        - Weather and time (e.g., “Is it hot today?”, “What season do you like?”)
        - Common-sense or trivia-type facts (e.g., “Why do leaves change color in fall?”, “What is the capital of France?”)
        - Personal preference or opinion style (e.g., “What’s your favorite food?”, “Do you like rainy days?”)
        - General curiosity (e.g., “How do airplanes fly?”, “Why is the sky blue?”)

        Make **half of the questions in English** and **half in Korean** (roughly 50 each).

        Each question must be short (one sentence) and *should not* ask for any detailed data, report content, business metrics, or document-based information.

        Return the result as a **valid JSON array** of objects.
        Each object must have:
        - "question": the question text
        - "needs_rag": 0

        Example objects:
        { "question": "What's your favorite movie?", "needs_rag": 0 }
        { "question": "하늘은 왜 파랗게 보여?", "needs_rag": 0 }

        Start your output with `[` and end with `]`, and include exactly **100 items**.
        DO NOT wrap the JSON in markdown code blocks. Return ONLY the raw JSON array.
        """
    else:
        prompt = """
    당신은 유용한 어시스턴트입니다.

    **외부 문서나 특정 보고서를 검색할 필요가 없는 100개의 서로 다른 짧은 질문**을 생성하세요.

    질문은 상식이나 일반 지식으로 답변할 수 있는 다양하고 간단한 일상 주제를 다루어야 합니다. 예를 들어:
    - 일상적인 대화 (예: 인사, 취미, 일상생활)
    - 날씨와 시간 (예: "오늘 날씨 어때?", "어떤 계절을 좋아해?")
    - 상식이나 상식 퀴즈 (예: "가을에 나뭇잎이 왜 색이 변해?", "프랑스의 수도는 어디야?")
    - 개인 선호나 의견 (예: "제일 좋아하는 음식이 뭐야?", "비 오는 날 좋아해?")
    - 일반적인 호기심 (예: "비행기는 어떻게 나는 거야?", "하늘은 왜 파래?")

    **모든 질문을 한국어로** 작성하세요.

    각 질문은 짧아야 하며(한 문장), 상세한 데이터, 보고서 내용, 업무 지표 또는 문서 기반 정보를 요구해서는 안 됩니다.

    결과를 **유효한 JSON 배열** 형식으로 반환하세요.
    각 객체는 다음을 포함해야 합니다:
    - "question": 질문 텍스트
    - "needs_rag": 0

    예시 객체:
    { "question": "제일 좋아하는 영화가 뭐야?", "needs_rag": 0 }
    { "question": "하늘은 왜 파랗게 보여?", "needs_rag": 0 }

    출력은 `[`로 시작하고 `]`로 끝나야 하며, 정확히 **100개의 항목**을 포함해야 합니다.
    마크다운 코드 블록으로 JSON을 감싸지 마세요. 순수한 JSON 배열만 반환하세요.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",     # 사용 모델명
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=2000
    )
    output = response.choices[0].message.content.strip()
    return output

def save_to_file(json_str, filepath="general_dataset.json"):
    # JSON 문자열이 올바른 배열 형태인지 검증
    data = json.loads(json_str)
    if not isinstance(data, list):
        raise ValueError("Output is not a JSON array")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(data)} items to {filepath}")

if __name__ == "__main__":
    option = 'Korean'
    json_str = generate_questions(option)
    save_to_file(json_str, f"./dataset/prepare_test_dataset/general_dataset({option}).json")