from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import json
import torch
import os
from tqdm import tqdm
import argparse
#export CUDA_VISIBLE_DEVICES=3

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

model_name = "Qwen/Qwen3-1.7B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# padding token 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# JSON 파일 불러오기
with open('/home/ljm/classifier/dataset/sample_train_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 커스텀 Dataset 클래스
class RAGDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        needs_rag = item['needs_rag']
        
        # 프롬프트 형식으로 변환
        prompt = f"질문: {question}\nRAG 필요 여부:"
        answer = f" {needs_rag}"
        
        # 전체 텍스트
        full_text = prompt + answer
        
        # 토크나이징
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # labels 생성 (input_ids와 동일하게 설정)
        labels = encodings['input_ids'].clone()
        
        # prompt 부분은 loss 계산에서 제외 (-100으로 설정)
        prompt_encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        prompt_length = prompt_encodings['input_ids'].shape[1]
        labels[:, :prompt_length] = -100
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }


if __name__ == '__main__':
    # 데이터셋 생성
    train_dataset = RAGDataset(data, tokenizer)

    # Trainer로 학습
    training_args = TrainingArguments(
        output_dir="./training_args",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        logging_steps=10,
        save_total_limit=2,
        learning_rate=2e-5,
        warmup_steps=100,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_dataset
    )

    # 학습 시작
    trainer.train()

    # 모델 저장
    model.save_pretrained("./trained_model/rag_classifier")
    tokenizer.save_pretrained("./trained_model/rag_classifier")