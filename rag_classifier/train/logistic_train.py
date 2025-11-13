import os
#export CUDA_VISIBLE_DEVICES=3

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# 데이터 준비
data = [
    {"question": "REST API란?", "needs_rag": 0},
    {"question": "Docker와 Kubernetes 차이점은?", "needs_rag": 0},
    {"question": "Q2 매출 보고서 내용 알려줘", "needs_rag": 1},
    {"question": "2023년 프로젝트 결과 요약해줘", "needs_rag": 1},
    {"question": "작년 연간 실적 보고서 있어?", "needs_rag": 1},
]

texts = [d["question"] for d in data]
labels = [d["needs_rag"] for d in data]

# Qwen 임베딩 모델 로드
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

# 문장 임베딩 생성
embeddings = model.encode(texts, normalize_embeddings=True)

# 학습/검증 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.3, random_state=42
)

# 간단한 분류기 (로지스틱 회귀)
clf = LogisticRegression()
clf.fit(X_train, y_train)

# 성능 평가
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# 새로운 입력 테스트
new_questions = [
    "올해 분기별 실적 보고서 요약해줘",
    "Python의 GIL은 뭐야?",
]

new_embeds = model.encode(new_questions, normalize_embeddings=True)
preds = clf.predict(new_embeds)

for q, p in zip(new_questions, preds):
    print(f"'{q}' → RAG 필요 여부: {p}")