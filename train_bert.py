import json
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
TRAIN_FILE = Path("data/bert/train.jsonl")
VALID_FILE = Path("data/bert/valid.jsonl")
OUTPUT_DIR = Path("./results")
class EmotionJSONLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=128):
        self.texts = []
        self.labels = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                self.texts.append(data["sentences"])
                self.labels.append(data["labels"])
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = np.mean(predictions == labels)
    return {'accuracy': acc}
def main():
    if not TRAIN_FILE.is_file() or not VALID_FILE.is_file():
        raise FileNotFoundError("Data files not found.")
    model_name = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=8
    )
    train_dataset = EmotionJSONLDataset(TRAIN_FILE, tokenizer)
    valid_dataset = EmotionJSONLDataset(VALID_FILE, tokenizer)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics
    )
    print("Starting BERT fine-tuning...")
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("Finished!")
if __name__ == "__main__":
    main()
