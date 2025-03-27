import os
from flask import Flask, request, abort
import ollama

# 導入Line Bot V3 SDK
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi
from linebot.v3.webhook import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent

# 引入配置
from config import (
    LINE_CHANNEL_ACCESS_TOKEN, 
    LINE_CHANNEL_SECRET, 
    OLLAMA_MODEL, 
    WEBHOOK_URL
)

# Flask Web應用
app = Flask(__name__)

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
        
        # 課程相關知識庫
        self.course_info = {
            "announcements": [],
            "assignments": {},
            "course_content": {}
        }

    def send_startup_message(self):
        """在應用程式啟動時發送訊息"""
        try:
            # 請替換 YOUR_USER_ID 為您的 Line 使用者 ID
            self.line_messaging_api.push_message(
                to='YOUR_USER_ID',
                messages=[{
                    'type': 'text', 
                    'text': '🤖 機器視覺課程助教機器人已啟動！\n系統已就緒，歡迎使用。\n目前的Ollama模型：' + self.ollama_model
                }]
            )
            print("啟動訊息已成功發送")
        except Exception as e:
            print(f"發送啟動訊息時發生錯誤: {e}")
    
    def add_announcement(self, announcement):
        """新增課程公告"""
        self.course_info["announcements"].append(announcement)
    
    def add_assignment(self, assignment_name, details):
        """新增作業詳情"""
        self.course_info["assignments"][assignment_name] = details
    
    def add_course_content(self, topic, content):
        """新增課程內容"""
        self.course_info["course_content"][topic] = content
    
    def generate_response(self, user_query):
        """使用Ollama生成回應"""
        try:
            response = ollama.chat(
                model=self.ollama_model,  # 使用配置的模型
                messages=[
                    {
                        'role': 'system', 
                        'content': '妳是機器視覺課程的助教機器人，專門回答學生的課程相關問題。'
                    },
                    {
                        'role': 'user', 
                        'content': user_query
                    }
                ]
            )
            return response['message']['content']
        
        except Exception as e:
            return f"生成回應時發生錯誤: {str(e)}"

# 初始化Bot
course_bot = CourseAssistantBot()

@app.route("/webhook", methods=['POST'])
def webhook():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    
    try:
        course_bot.handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    
    return 'OK'

@course_bot.handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_query = event.message.text
    print(f"訊息: {user_query}")
    response = course_bot.generate_response(user_query)
    
    try:
        course_bot.line_messaging_api.reply_message(
            replyToken=event.replyToken,
            messages=[{'type': 'text', 'text': response}]
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

    # 在啟動時發送訊息
    course_bot.send_startup_message()

    app.run(port=5000)