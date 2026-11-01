# -*- coding: utf-8 -*-
"""
infer_bert.py

載入微調後的最佳 BERT 多元分類器，
對指定的中文對話進行 8 種情感類別的分類預測。
"""

from pathlib import Path
from pprint import pprint
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

# 指向模型儲存的路徑
MODEL_DIR = Path("./results")

# 題目指定預測的 5 個對話句子
TEST_TEXTS = [
    "我每天都能跟她一起上學，我好開心！",
    "最好的朋友要離開臺灣了，以後可能不容易再見面...",
    "我覺得我快不行了...",
    "剛剛收到研究所錄取的通知書！",
    "今年的冬天好像比較晚來。",
    "我覺得今天天氣很好！",
    "這也太誇張了吧...",
]

def main():
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"找不到 fine-tuned model：{MODEL_DIR.resolve()}\n"
            "請先確認是否已執行 train_bert.py 完成模型微調。"
        )

    # 載入模型與 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))

    # 自動配對硬體 (GPU/CPU)
    device = 0 if torch.cuda.is_available() else -1
    print("Inference device：" + (torch.cuda.get_device_name(0) if device == 0 else "CPU"))
    print("-" * 72)

    # 建立多文本分類 Pipeline
    classifier = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    # 執行推論
    results = classifier(TEST_TEXTS, truncation=True, max_length=512)

    # 手動建立情緒對照表，百分之百避開 Windows 終端機 (CMD) 的寫入編碼錯誤
    # 根據 Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset 資料集的標準排序定義
    mapping = {
        "LABEL_0": "厭惡語調",
        "LABEL_1": "平淡語氣",
        "LABEL_2": "悲傷語調",
        "LABEL_3": "憤怒語調",
        "LABEL_4": "疑問語調",
        "LABEL_5": "開心語調",
        "LABEL_6": "驚奇語調",
        "LABEL_7": "驚訝語調"
    }

    # 顯示結果 (完全比照作業要求格式與小數點後 4 位)
    print("\n" + "="*30 + " 預測結果展示 " + "="*30)
    for text, result in zip(TEST_TEXTS, results):
        label_str = result['label']
        # 從對照表找出對應的中文，如果找不到則顯示原本的 LABEL 代號
        chinese_label = mapping.get(label_str, label_str)
        score = result['score']
        print(f"{text} => {chinese_label} ({score:.4f})")
    print("=" * 74)

    print("Raw pipeline output：")
    pprint(results)

if __name__ == "__main__":
    main()
