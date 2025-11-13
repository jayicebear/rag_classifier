from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json 

# GPU 설정 (CUDA_VISIBLE_DEVICES 환경변수는 외부에서 설정 가능)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 학습된 모델 경로
model_path = "./trained_model/rag_classifier"

#  모델 & 토크나이저 로드
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

#  테스트용 질문 리스트
# JSON 파일 불러오기
with open('/home/ljm/classifier/dataset/sample_test_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


if __name__ == '__main__':
    # 각 질문에 대해 프롬프트 구성
    predictions = []

    for question in data:
        prompt = f"질문: {question['question']}\nRAG 필요 여부:"
        
        # 토크나이즈
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # 모델 생성 (추론)
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,     # 0 또는 1만 생성하도록 제한
            do_sample=False,     # deterministic output
            pad_token_id=tokenizer.eos_token_id
        )
        
        # 결과 디코딩
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 숫자 추출
        # 예: "질문: Q2 매출 보고서 내용 알려줘\nRAG 필요 여부: 1" → "1" 추출
        if "RAG 필요 여부:" in result:
            pred = result.split("RAG 필요 여부:")[-1].strip()
        else:
            pred = result.strip()
        
        # 후처리 (0 또는 1 이외 문자열 제거)
        pred = ''.join([c for c in pred if c in ['0','1']]) or "?"
        
        predictions.append(pred)
        
        print(f"[입력] {question['question']}\n→ 예측 결과: {pred}\n")
        
    output_path = "./test_result/predictions.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
