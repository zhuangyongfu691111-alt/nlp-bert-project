# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import chromadb
from sentence_transformers import SentenceTransformer
def crawl_health_news():
    print("Starting health news web crawler...")
    news_data = []
    dummy_titles = [
        "秋天皮膚乾癢怎麼辦？醫教保濕時機、洗澡與止癢重點",
        "腎友如何護腎又保肌？飲食選擇、醫療介入、運動策略一次掌握",
        "跑步後疼痛不一定是拉傷！專家解析「肌力訓練」為何重要 3點防運動傷害",
        "外食怎麼吃才不傷腎？營養師揭「5大原則」這些醬料、湯汁要少碰",
        "不用大老遠跑醫院！「遠距會診」就能獲專科評估 串起社區與醫院照護"
    ]
    for i, t in enumerate(dummy_titles, 1):
        news_data.append({
            "流水號": i,
            "新聞標題": t,
            "新聞連結": f"https://healthnews.com.tw{10000+i}",
            "文章全文": f"這是關於「{t}」的完整健康醫學分析與專家建議全文指南。",
            "文章摘要": f"【地端摘要】本篇報導聚焦於「{t}」，指出民眾在日常生活中應建立正確的保健習慣。",
            "文章新增時間": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    return pd.DataFrame(news_data)
def main():
    df = crawl_health_news()
    print("\n--- Data table created successfully! ---")
    print(df[["流水號", "新聞標題"]])
    print("-" * 50)
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    try:
        chroma_client.delete_collection(name="health_news")
    except:
        pass
    collection = chroma_client.get_or_create_collection(name="health_news")
    print("Loading multilingual local Embedding model...")
    embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    print("Converting summaries into vectors and saving index...")
    for _, row in df.iterrows():
        embedding = embed_model.encode(row["文章摘要"]).tolist()
        metadata = {
            "title": str(row["新聞標題"]),
            "link": str(row["新聞連結"]),
            "time": str(row["文章新增時間"])
        }
        collection.add(
            embeddings=[embedding],
            documents=[row["文章全文"]],
            metadatas=[metadata],
            ids=[str(row["流水號"])]
        )
    print("Finished! Vector database index saved in ./chroma_db")
if __name__ == "__main__":
    main()
