# 온프레미스 RAG 라우팅 분류기 

## 프로젝트 개요
사용자가 입력한 질문이 **일반적인 일상 대화**인지, 아니면 **업로드된 문서(PDF 등)에 대한 전문 질문**인지를 자동으로 판별하여  
- 일상 대화 → 사전학습된 지식(pretrained)으로 답변  
- 문서 관련 질문 → 업로드된 문서 청크 기반 RAG로 답변  

하도록 라우팅하는 **온프레미스 분류기** 구축.  
오픈소스 LLM만 사용하며, 모든 데이터와 모델은 로컬 환경에서 실행됩니다.

## 데이터 준비 및 Fine-tuning 과정

```text
PDF → Docling → Markdown → MarkdownHeaderTextSplitter → Chunks
                                    ↓
                             GPT-4o API로 청크당 3개 질문 생성
                                    ↓
                     {question: "...", chunk: "..."} 페어 1만+건
                                    ↓
                           Llama-3.1-8B / Mistral-7B 등 LLM Fine-tuning
                                    ↓
                           온프레미스 라우팅 분류기 완성
