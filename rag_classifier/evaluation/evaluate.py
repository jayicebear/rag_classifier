from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import json

# JSON 파일 불러오기
prediction_file_name = "predictions"
answer_file_name = "sample_answer"
with open(f'/home/ljm/classifier/test_result/{prediction_file_name}.json', 'r', encoding='utf-8') as f:
    prediction = json.load(f)
# JSON 파일 불러오기
with open(f'/home/ljm/classifier/test_result/{answer_file_name}.json', 'r', encoding='utf-8') as f:
    answer = json.load(f)

answer = [int(x) for x in answer]
prediction = [int(x) for x in prediction]

metrics = {
    "accuracy": accuracy_score(answer, prediction),
    "precision": precision_score(answer, prediction, zero_division=0),
    "recall": recall_score(answer, prediction, zero_division=0),
    "f1_score": f1_score(answer, prediction, zero_division=0)
}

# 2️⃣ 세부 리포트 (각 클래스별 precision/recall 포함)
report = classification_report(answer, prediction, output_dict=True)
metrics["detailed_report"] = report

# 3️⃣ 결과 저장
with open("./evaluation_report/report.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("✅ 평가 완료! 결과가 'metrics.json' 파일로 저장되었습니다.")
print(json.dumps(metrics, indent=2, ensure_ascii=False))