# -*- coding: utf-8 -*-
"""
plot_bert_metrics.py

自動搜尋結果目錄（包含 checkpoint 子資料夾）中的 trainer_state.json，
並繪製輸出 Loss 曲線與 Accuracy 曲線。
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")  # 使用非 GUI 後端，避免在終端機環境中跳出視窗報錯

import matplotlib.pyplot as plt

# =============================================================================
# 1. 路徑與檔案自動搜尋設定
# =============================================================================
OUTPUT_DIR = Path("./results")
LOSS_PLOT_FILE = OUTPUT_DIR / "loss_curve.png"
ACC_PLOT_FILE = OUTPUT_DIR / "accuracy_curve.png"

def find_trainer_state(base_dir: Path):
    """自動在 base_dir 及其所有子目錄（如 checkpoint-xxx）中尋找 trainer_state.json"""
    # 優先檢查最外層
    direct_file = base_dir / "trainer_state.json"
    if direct_file.is_file():
        return direct_file
        
    # 遍歷子目錄尋找（例如 results/checkpoint-624/trainer_state.json）
    for path in base_dir.rglob("trainer_state.json"):
        if path.is_file():
            return path
            
    return None

# =============================================================================
# 2. 主流程
# =============================================================================
def main():
    state_file = find_trainer_state(OUTPUT_DIR)
    
    if state_file is None:
        raise FileNotFoundError(
            f"找不到任何 trainer_state.json 檔案（檢查了 {OUTPUT_DIR.resolve()} 及其子目錄）。\n"
            "請確認 train_bert.py 訓練是否確實完整結束並產生了 checkpoint 檔案。"
        )

    print(f"成功找到訓練日誌紀錄：{state_file.resolve()}")

    with state_file.open("r", encoding="utf-8") as file:
        state = json.load(file)

    logs = state.get("log_history", [])

    if not logs:
        raise RuntimeError("trainer_state.json 中沒有 log_history。")

    # -------------------------------------------------------------------------
    # Training / Validation Loss 曲線繪製
    # -------------------------------------------------------------------------
    train_loss = [
        (item["step"], item["loss"])
        for item in logs
        if "step" in item and "loss" in item
    ]

    valid_loss = [
        (item["step"], item["eval_loss"])
        for item in logs
        if "step" in item and "eval_loss" in item
    ]

    if train_loss or valid_loss:
        plt.figure(figsize=(10, 6))

        if train_loss:
            x, y = zip(*train_loss)
            plt.plot(x, y, label="Training Loss", linewidth=1.8)

        if valid_loss:
            x, y = zip(*valid_loss)
            plt.plot(x, y, marker="o", label="Validation Loss", linewidth=1.8)

        plt.xlabel("Optimizer Step")
        plt.ylabel("Loss")
        plt.title("BERT Multi-Emotion Training and Validation Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(LOSS_PLOT_FILE, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Loss 圖表已成功儲存：{LOSS_PLOT_FILE.resolve()}")

    # -------------------------------------------------------------------------
    # Validation Accuracy 曲線繪製
    # -------------------------------------------------------------------------
    eval_acc = [
        (item["step"], item["eval_accuracy"])
        for item in logs
        if "step" in item and "eval_accuracy" in item
    ]

    if eval_acc:
        plt.figure(figsize=(10, 6))

        x, y = zip(*eval_acc)
        plt.plot(x, y, marker="o", linewidth=1.8, color="orange", label="Validation Accuracy")

        plt.xlabel("Optimizer Step")
        plt.ylabel("Accuracy")
        plt.title("BERT Multi-Emotion Validation Accuracy")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(ACC_PLOT_FILE, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Accuracy 圖表已成功儲存：{ACC_PLOT_FILE.resolve()}")

    print("-" * 50)
    print(f"Training loss points: {len(train_loss)}")
    print(f"Validation loss points: {len(valid_loss)}")
    print(f"Validation Accuracy points: {len(eval_acc)}")

if __name__ == "__main__":
    main()
