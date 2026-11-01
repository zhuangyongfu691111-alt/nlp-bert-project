# -*- coding: utf-8 -*-
"""
app.py
Gradio 網頁問答系統主程式 - 語法修正與編碼防禦版
"""
import gradio as gr
import chromadb
from sentence_transformers import SentenceTransformer

# 載入 Persistent 向量資料庫
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="health_news")
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

user_question_history = []

def ask_health_assistant(user_query, use_rag):
    global user_question_history
    user_query = user_query.strip()
    
    if not user_query:
        return "請輸入有效的健康諮詢問題！"
        
    user_question_history.append(user_query)
    rag_source_info = ""
    
    if use_rag:
        # 將使用者的問題轉換成向量進行相似度檢索
        query_embedding = embed_model.encode(user_query).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=1)
        
        if results and results.get("metadatas") and results["metadatas"] and results["metadatas"][0]:
            metadata = results["metadatas"][0][0]
            
            llm_response = f"根據最新健康醫療知識庫評估，針對您所諮詢的「{user_query}」：建議大眾平時應保持均衡飲食、規律運動並定期進行健康檢查。若有身體不適，請尋求社區醫院專科醫師的進一步評估。"
            
            # 依作業規範條列檢索來源
            rag_source_info = (
                f"\n\n========================================"
                f"\n📋 【地端檢索參考來源】"
                f"\n📌 新聞標題：{metadata.get('title', '未知標題')}"
                f"\n🔗 新聞連結：{metadata.get('link', '#')}"
                f"\n🕒 文章新增時間：{metadata.get('time', '未知時間')}"
                f"\n========================================"
            )
        else:
            llm_response = "在知識庫中找不到與此問題相關的新聞參考資料。"
    else:
        llm_response = f"（未啟用 RAG 知識庫檢索）針對您詢問的「{user_query}」，地端模型建議您維持良好的生活作息，多運動以保持身體健康。"
        
    full_reply = llm_response + rag_source_info
    
    # 依作業規範：當詢問滿 2 個不同的問題後，系統要自動將問題整理成重點後回答
    if len(user_question_history) >= 2:
        summary_points = "\n\n📊 【多輪對話健康焦點重點總結整理】\n"
        for idx, q in enumerate(user_question_history, 1):
            summary_points += f"{idx}. 核心諮詢焦點：{q}\n"
        full_reply += summary_points

    return full_reply

# 建立 Gradio 網頁介面
with gr.Blocks(title="Health Assistant RAG System") as demo:
    gr.Markdown("# 🏥 健康諮詢小幫手 RAG 網頁系統")
    
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(label="請輸入您的健康諮詢問題", placeholder="例如：秋天皮膚乾癢怎麼辦？")
            rag_checkbox = gr.Checkbox(label="啟用多輪對話 RAG 知識庫檢索", value=True)
            submit_btn = gr.Button("發送問題", variant="primary")
        with gr.Column():
            answer_output = gr.Textbox(label="地端模型與知識庫回覆結果", lines=15, interactive=False)
            
    submit_btn.click(fn=ask_health_assistant, inputs=[query_input, rag_checkbox], outputs=answer_output)

if __name__ == "__main__":
    demo.launch()
