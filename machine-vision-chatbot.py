import os
from flask import Flask, request, abort
import ollama
import time
import re
from datetime import datetime, timedelta, timezone

# 導入Line Bot V3 SDK
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi
from linebot.v3.webhook import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.messaging.models import TextMessage, ReplyMessageRequest, PushMessageRequest, BroadcastRequest

# 新增公告分塊與檢索
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

import json


###################################################################

# 引入配置
from config import (
    LINE_CHANNEL_ACCESS_TOKEN, 
    LINE_CHANNEL_SECRET, 
    OLLAMA_MODEL, 
    WEBHOOK_URL
)

# Flask Web應用
app = Flask(__name__)

def get_taiwan_time():
    """取得台灣時間 (GMT+8)"""
    # 創建台灣時區 (UTC+8)
    tw_timezone = timezone(timedelta(hours=8))
    # 獲取目前UTC時間並轉換為台灣時間
    tw_time = datetime.now(tw_timezone)
    # 格式化時間字串
    print(f"tw_time:{tw_time}")
    print(f"現在時間:{datetime.now()}")
    return tw_time.strftime("%Y-%m-%d %H:%M:%S")

system_prompt = f"""You are the Teaching Assistant (TA) chatbot for the Machine Vision Course. You are only allowed to provide course announcements, syllabus topics, and assignment guidelines. You must answer only based on the provided information below and must not answer any questions beyond this data. You are strictly prohibited from giving any form of code or logic.
                        
                        You are strictly forbidden from:
                        Writing any code (e.g., Python, C++, MATLAB, etc.)
                        Providing any functions, algorithmic logic, steps, or principles
                        Explaining code, analyzing logic, or suggesting alternate implementations
                        Answering questions such as “how to implement,” “what to do,” or “what if I can’t use a certain function”
                        Even if the user paraphrases, indirectly asks, or only requests the “logic,” you still may not respond

                        For such questions, your only allowed replies are one of the following:
                        “I cannot answer.”
                        “I cannot provide.”
                        “This is beyond my responsibility. Please email the TA for help.”
                        “I cannot provide assignment solutions.”

                        You are allowed to answer:
                        This week’s lecture topics and summary (only if explicitly mentioned in announcements; otherwise say: “I cannot answer, please email the TA for help.”)
                        Course announcements, deadlines, and submission methods
                        Assignment content descriptions (verbatim or summarized from the announcements)
                        Assignment rules (allowed packages, restrictions, file formats, etc.)

                        Rules:
                        All responses must be brief (within 50 words)
                        Even if asked repeatedly, you must not provide technical explanations or code
                        Your role is to act as a TA chatbot to prevent students from copying homework or getting models to complete code for them, while still answering questions about the course.
"""

class CourseAssistantBot:
    def __init__(self):
        # V3 SDK 配置
        configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
        
        # 創建API客戶端
        self.line_api_client = ApiClient(configuration)
        self.line_messaging_api = MessagingApi(self.line_api_client)
        
        # Webhook Handler
        self.handler = WebhookHandler(LINE_CHANNEL_SECRET)
        
        # 使用配置中的Ollama模型
        self.ollama_model = OLLAMA_MODEL
        print(f"OLLAMA_MODEL: {OLLAMA_MODEL}")
        
        # 課程相關知識庫
        self.course_info = {
            "announcements": [],
            "assignments": {},
            "course_content": {}
        }

        self.index = None
        self.embedder = None
        self.data_text = None
        self.data_chunks = []

    def send_startup_message(self):
        """在應用程式啟動時發送訊息"""
        try:
            # 使用broadcast方法發送給所有好友
            broadcast_request = BroadcastRequest(
                messages=[TextMessage(
                    type='text',
                    text='🤖 機器視覺課程助教機器人已啟動！\n台灣時間: ' + get_taiwan_time() + '\n目前Ollama模型: ' + self.ollama_model
                )]
            )
            
            # 發送訊息
            self.line_messaging_api.broadcast(broadcast_request)
            print("啟動訊息已成功廣播")
        except Exception as e:
            print(f"發送啟動訊息時發生錯誤: {e}")
            print("==================================================")
    
    def load_data(self):
        with open('vision_course_announcements_eng.txt', 'r', encoding='utf-8') as f:
            self.data_text = f.read()

        # pattern = r'(?=(公告\s\d+【\d{4}/\d{2}/\d{2}】：|作業[一二三]：))'
        pattern = r'(?=(Announcement\s\d\s+[\d{4}/\d{2}/\d{2}]:|Assignment[123]:))'
        # 先用 re.split 拆分，會保留分隔符作為元素
        chunks = re.split(pattern, self.data_text)

        self.data_chunks = []
        for i in range(1, len(chunks), 2):
            self.data_chunks.append(chunks[i+1])

        # 建立嵌入模型
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        batch_size = 32  # 根據 GPU 記憶體調整（從 32 開始嘗試）
        chunk_embeddings = []

        for i in range(0, len(self.data_chunks), batch_size):
            batch = self.data_chunks[i:i+batch_size]
            embeddings = self.embedder.encode(batch, convert_to_numpy=True, show_progress_bar=True)
            chunk_embeddings.append(embeddings)

        chunk_embeddings = np.concatenate(chunk_embeddings, axis=0)


        # 建立FAISS索引
        self.index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
        self.index.add(chunk_embeddings)

    def add_announcement(self, announcement):
        """新增課程公告"""
        self.course_info["announcements"].append(announcement)
    
    def add_assignment(self, assignment_name, details):
        """新增作業詳情"""
        self.course_info["assignments"][assignment_name] = details
    
    def add_course_content(self, topic, content):
        """新增課程內容"""
        self.course_info["course_content"][topic] = content
    
    def retrieve_law_context(self, question, top_k=10):
        q_emb = self.embedder.encode([question], convert_to_numpy=True)
        D, I = self.index.search(q_emb, top_k)
        return [self.data_chunks[idx] for idx in I[0]]

    def generate_prompt(self, user_input):
        prompt = f"""{system_prompt}

                    ** Here is the only data you can reference. You must not use any outside knowledge or inference **:
                    Class time: Mondays 10:00–12:00 and Wednesdays 16:00–17:00
                    {self.data_text}

                    =====================================================================
                    ** Today's date: {get_taiwan_time()}, Please provide answers based on this date. **
                    The following are students' questions. Please respond briefly (within 50 words) based on the guidelines above:
                    {user_input}
                """
        return prompt
    
    def generate_response(self, user_query):
        """使用Ollama生成回應"""
        try:
            # assistant_prompts = "\n\n".join(self.retrieve_law_context(user_query, top_k=5)) # use RAG
            # print(f"assistant_prompts:{assistant_prompts}")
            prompt = self.generate_prompt(user_query)
            messages =  [{"role": "user", "content": prompt}]
            response = ollama.chat(model=self.ollama_model, messages=messages)
            # print(f"原始回應: {response['message']['content']}\n")
            response = re.sub(r'^\n+', '', response['message']['content'], flags=re.DOTALL)
            return response
        
        except Exception as e:
            return f"生成回應時發生錯誤: {str(e)}"

# 初始化Bot
course_bot = CourseAssistantBot()

@app.route("/webhook", methods=['POST'])
def webhook():
    # print("進去webhook了")
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    # print(f"收到webhook請求: {body}")
    try:
        course_bot.handler.handle(body, signature)
    except InvalidSignatureError:
        print("簽名驗證失敗")
        abort(400)
    except Exception as e:
        print(f"處理webhook時發生錯誤: {e}")
    
    return 'OK'

@app.route("/test", methods=['GET'])
def test():
    return "機器視覺課程助教機器人運行中！"

@course_bot.handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_query = event.message.text
    user_id = event.source.user_id
    now = datetime.now().strftime("%H:%M")
    print(f"{user_id} | {now} 傳送訊息: {user_query}")
    response = course_bot.generate_response(user_query)
    now = datetime.now().strftime("%H:%M")
    print(f"機器人回覆 {now}: {response}")
    
    # replyToken=event.reply_token
    # messages=TextMessage(text=response)
    try:
        course_bot.line_messaging_api.reply_message(
            ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[
            TextMessage(text=response)
            ]
        )
            )
        
    except Exception as e:
        print(f"Reply message error: {e}")

# 初始化範例數據
def init_course_data():
    # 新增公告
    course_bot.add_announcement("第一次作業將於下週一發布，請同學們準備")
    course_bot.add_announcement("課程期中專案主題已公布，請同學們盡快開始準備")
    
    # 新增作業詳情
    course_bot.add_assignment("作業一", {
        "名稱": "影像分類任務",
        "截止日期": "2024-04-15",
        "要求": "使用CNN實現CIFAR-10資料集分類",
        "評分標準": {
            "模型準確率": 0.4,
            "程式碼規範": 0.3,
            "報告文檔": 0.3
        }
    })
    
    # 新增課程內容
    course_bot.add_course_content("CNN架構", "介紹卷積神經網路的基本原理和實作")
    course_bot.add_course_content("影像前處理", "數據增強、標準化和正規化技術")

if __name__ == '__main__':
    init_course_data()

    course_bot.load_data() # 載入RAG

    # 在啟動時發送訊息
    # course_bot.send_startup_message()

    app.run(host="0.0.0.0", port=5000)