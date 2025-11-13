import json

# 학습 데이터 샘플
train_data = [
    # 일상 대화 (needs_rag: 0)
    {"question": "안녕하세요", "needs_rag": 0},
    {"question": "안녕", "needs_rag": 0},
    {"question": "오늘 날씨 어때?", "needs_rag": 0},
    {"question": "점심 뭐 먹지?", "needs_rag": 0},
    {"question": "고마워", "needs_rag": 0},
    {"question": "잘 지내?", "needs_rag": 0},
    {"question": "내일 뭐해?", "needs_rag": 0},
    {"question": "피곤해", "needs_rag": 0},
    {"question": "주말 잘 보냈어?", "needs_rag": 0},
    {"question": "시간이 몇 시야?", "needs_rag": 0},
    {"question": "좋은 아침", "needs_rag": 0},
    {"question": "잘자", "needs_rag": 0},
    {"question": "오늘 기분 어때?", "needs_rag": 0},
    {"question": "커피 한잔 할래?", "needs_rag": 0},
    {"question": "너 이름이 뭐야?", "needs_rag": 0},
    
    # 일반 상식 질문 (needs_rag: 0)
    {"question": "파이썬으로 리스트 정렬하는 방법은?", "needs_rag": 0},
    {"question": "한국의 수도는?", "needs_rag": 0},
    {"question": "지구는 몇 개의 대륙이 있어?", "needs_rag": 0},
    {"question": "1+1은 뭐야?", "needs_rag": 0},
    {"question": "머신러닝이 뭐야?", "needs_rag": 0},
    {"question": "for 루프 예제 좀 알려줘", "needs_rag": 0},
    {"question": "SQL JOIN 종류 설명해줘", "needs_rag": 0},
    {"question": "React hooks 사용법 알려줘", "needs_rag": 0},
    {"question": "REST API란?", "needs_rag": 0},
    {"question": "Docker와 Kubernetes 차이점은?", "needs_rag": 0},
    
    # 업무 관련 - 문서 검색 필요 (needs_rag: 1)
    {"question": "Q2 매출 보고서 내용 알려줘", "needs_rag": 1},
    {"question": "2023년 프로젝트 결과 요약해줘", "needs_rag": 1},
    {"question": "작년 연간 실적 보고서 있어?", "needs_rag": 1},
    {"question": "고객 계약서 양식 좀 찾아줘", "needs_rag": 1},
    {"question": "지난주 회의록 내용 확인해줘", "needs_rag": 1},
    {"question": "인사 규정 문서 어디 있지?", "needs_rag": 1},
    {"question": "이번 달 예산안 승인 받았어?", "needs_rag": 1},
    {"question": "A 프로젝트 진행 현황 알려줘", "needs_rag": 1},
    {"question": "마케팅 전략 문서 보내줘", "needs_rag": 1},
    {"question": "신제품 기획서 완성됐어?", "needs_rag": 1},
    {"question": "지난 분기 영업 실적은?", "needs_rag": 1},
    {"question": "B사 제안서 내용 정리해줘", "needs_rag": 1},
    {"question": "연구개발 로드맵 문서 찾아줘", "needs_rag": 1},
    {"question": "직원 복지 규정 확인하고 싶어", "needs_rag": 1},
    {"question": "내부 감사 보고서 결과는?", "needs_rag": 1},
    {"question": "IT 시스템 운영 매뉴얼 있어?", "needs_rag": 1},
    {"question": "고객사 미팅 자료 준비됐어?", "needs_rag": 1},
    {"question": "품질 관리 프로세스 문서 보여줘", "needs_rag": 1},
    {"question": "2024년 사업 계획서 내용은?", "needs_rag": 1},
    {"question": "경쟁사 분석 보고서 요약해줘", "needs_rag": 1},
    
    # 추가 에지 케이스
    {"question": "우리 회사 설립일은?", "needs_rag": 1},
    {"question": "CEO 인사말 내용 알려줘", "needs_rag": 1},
    {"question": "직원 교육 일정 확인해줘", "needs_rag": 1},
    {"question": "보안 정책 문서 어디 있어?", "needs_rag": 1},
    {"question": "출장비 청구 방법 알려줘", "needs_rag": 1},
]

# 테스트 데이터 샘플
test_data = [
    # 일상 대화
    {"question": "안녕하세요 반갑습니다", "needs_rag": 0},
    {"question": "오늘 저녁 뭐 먹을까?", "needs_rag": 0},
    {"question": "감사합니다", "needs_rag": 0},
    {"question": "화이팅!", "needs_rag": 0},
    {"question": "내일 날씨 좋을까?", "needs_rag": 0},
    
    # 일반 상식
    {"question": "딥러닝과 머신러닝 차이는?", "needs_rag": 0},
    {"question": "Git merge와 rebase 차이점", "needs_rag": 0},
    {"question": "클라우드 컴퓨팅이란?", "needs_rag": 0},
    {"question": "파이썬 딕셔너리 사용법", "needs_rag": 0},
    {"question": "HTTP와 HTTPS 차이", "needs_rag": 0},
    
    # 업무 문서 검색 필요
    {"question": "Q3 재무제표 확인해줘", "needs_rag": 1},
    {"question": "지난달 회의록 있어?", "needs_rag": 1},
    {"question": "신입사원 온보딩 자료 찾아줘", "needs_rag": 1},
    {"question": "C 프로젝트 타임라인 알려줘", "needs_rag": 1},
    {"question": "올해 채용 계획 문서는?", "needs_rag": 1},
    {"question": "고객 만족도 조사 결과 요약", "needs_rag": 1},
    {"question": "기술 스택 문서 보고 싶어", "needs_rag": 1},
    {"question": "부서별 KPI 목표치는?", "needs_rag": 1},
    {"question": "재택근무 지침 확인하고 싶어", "needs_rag": 1},
    {"question": "협력사 계약 조건 알려줘", "needs_rag": 1},
]


answer_list = []
for i in test_data:
    answer_list.append(i['needs_rag'])
    
with open('./test_result/sample_answer.json', 'w', encoding='utf-8') as f:
    json.dump(answer_list, f, ensure_ascii=False, indent=2)

# JSON 파일로 저장
# with open('./dataset/sample_train_data.json', 'w', encoding='utf-8') as f:
#     json.dump(train_data, f, ensure_ascii=False, indent=2)

# with open('./dataset/sample_test_data.json', 'w', encoding='utf-8') as f:
#     json.dump(test_data, f, ensure_ascii=False, indent=2)

    
    
# print(f"✓ train_data.json 생성 완료 ({len(train_data)} samples)")
# print(f"✓ test_data.json 생성 완료 ({len(test_data)} samples)")
# print("\n학습 데이터 분포:")
# print(f"  - needs_rag=0 (일상/상식): {sum(1 for x in train_data if x['needs_rag'] == 0)}개")
# print(f"  - needs_rag=1 (업무/문서): {sum(1 for x in train_data if x['needs_rag'] == 1)}개")
# print("\n테스트 데이터 분포:")
# print(f"  - needs_rag=0 (일상/상식): {sum(1 for x in test_data if x['needs_rag'] == 0)}개")
# print(f"  - needs_rag=1 (업무/문서): {sum(1 for x in test_data if x['needs_rag'] == 1)}개")