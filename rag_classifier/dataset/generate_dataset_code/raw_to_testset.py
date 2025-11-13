import json

# 원본 JSON 파일 읽기
file_name = 'safe_sql'
language = 'Korean'
option = 'Rag'
input_file = f"/home/ljm/classifier/dataset/chunk_and_question({language})/{file_name}_chunk_questions.json"
if option == 'Rag' or 'RAG':
    output_file = f"/home/ljm/classifier/dataset/prepare_test_dataset/{file_name}_need_rag_dataset.json"
else: 
    output_file = f"/home/ljm/classifier/dataset/prepare_test_dataset/{file_name}_X_need_rag_dataset.json"
    
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 새로운 형식으로 변환
result = []

for item in data:
    # question이 리스트인 경우 각각을 개별 항목으로 변환
    if isinstance(item.get('question'), list):
        for question in item['question']:
            result.append({
                "question": question,
                "needs_rag": 1  # chunk_and_question 데이터는 RAG가 필요한 질문
            })
    # question이 단일 문자열인 경우
    elif isinstance(item.get('question'), str):
        result.append({
            "question": item['question'],
            "needs_rag": 1
        })

# 결과 저장
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f" 변환 완료!")
print(f"원본 chunk 개수: {len(data)}")
print(f"변환된 question 개수: {len(result)}")
print(f"저장 위치: {output_file}")

# 샘플 출력
print("\n=== 샘플 데이터 (처음 3개) ===")
for i, item in enumerate(result[:3]):
    print(f"{i+1}. {item}")