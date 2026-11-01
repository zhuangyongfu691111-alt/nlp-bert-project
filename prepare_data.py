"""
prepare_data.py

將 Chinese_Multi-Emotion_Dialogue_Dataset 轉成 
BERT 多元分類訓練所需的 train / validation JSONL。
"""

import json
from collections import Counter
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# 資料與路徑設定
OUTPUT_DIR = Path("data/bert")
VALID_RATIO = 0.2
SEED = 42

def load_hf_emotion_dataset():
    print("正在載入 Chinese_Multi-Emotion_Dialogue_Dataset 資料集...")
    try:
        # 使用官方 pandas 方法直接線上讀取
        df = pd.read_csv("hf://datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset/data.csv")
    except Exception as e:
        print(f"線上載入失敗 ({e})，嘗試讀取本地 data.csv...")
        if Path("data.csv").is_file():
            df = pd.read_csv("data.csv")
        else:
            raise FileNotFoundError("找不到資料來源，請確保網路連線正常或本地有 data.csv 檔案。")

    # 移除空值並清理空白文字
    df = df.dropna(subset=['text', 'emotion'])
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'] != ""]

    # 建立 8 種情緒的中文與 ID 映射字典
    unique_emotions = sorted(df['emotion'].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(unique_emotions)}
    
    print("\n--- 偵測到以下 8 種情緒標籤映射 ---")
    for label, idx in label2id.items():
        print(f"  {label} -> {idx}")
    print("-" * 40)

    texts = df['text'].tolist()
    labels = df['emotion'].map(label2id).tolist()
    
    return texts, labels, unique_emotions

def save_jsonl(texts, labels, path: Path):
    with path.open("w", encoding="utf-8") as file:
        for text, label in zip(texts, labels):
            row = {"sentences": text, "labels": label}
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

def print_label_distribution(name, labels, unique_emotions):
    counts = Counter(labels)
    total = len(labels)
    print(f"{name}：{total} 筆")
    for label_idx in sorted(counts):
        count = counts[label_idx]
        ratio = count / total
        label_name = unique_emotions[label_idx]
        print(f"  label={label_idx} ({label_name}): {count} 筆 ({ratio:.2%})")

def main():
    texts, labels, unique_emotions = load_hf_emotion_dataset()

    print("=" * 72)
    print_label_distribution("全部資料", labels, unique_emotions)
    print("=" * 72)

    # 以標籤進行分層抽樣切分 (Stratified Split)
    train_texts, valid_texts, train_labels, valid_labels = train_test_split(
        texts, labels, test_size=VALID_RATIO, random_state=SEED, stratify=labels
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / "train.jsonl"
    valid_path = OUTPUT_DIR / "valid.jsonl"

    save_jsonl(train_texts, train_labels, train_path)
    save_jsonl(valid_texts, valid_labels, valid_path)

    print_label_distribution("Training", train_labels, unique_emotions)
    print("-" * 40)
    print_label_distribution("Validation", valid_labels, unique_emotions)
    print("=" * 72)
    print(f"Training JSONL 已產生：{train_path}")
    print(f"Validation JSONL 已產生：{valid_path}")
    print("=" * 72)

if __name__ == "__main__":
    main()
