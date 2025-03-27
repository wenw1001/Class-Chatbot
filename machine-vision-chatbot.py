import os
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import LineBotApiError, InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import ollama
from flask import Flask, request, abort
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# Line Bot 設定
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')

# Ollama 語言模型設定
class CourseAssistantBot:
    def __init__(self):
        # Line Bot 初始化
        self.line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
        self.handler = WebhookHandler(LINE_CHANNEL_SECRET)
        
        # 課程相關知識庫
        self.course_info = {
            "announcements": [],
            "assignments": {},
            "course_content": {}
        }
    
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
            # 結合課程知識庫和語言模型
            context = f"""
            課程知識庫:
            公告: {self.course_info['announcements']}
            作業: {self.course_info['assignments']}
            課程內容: {self.course_info['course_content']}
            
            使用者問題: {user_query}
            """
            
            response = ollama.chat(
                model='llama2',  # 可以根據需要替換模型
                messages=[
                    {
                        'role': 'system', 
                        'content': '妳是機器視覺課程的助教機器人，專門回答學生的課程相關問題。'
                    },
                    {
                        'role': 'user', 
                        'content': context
                    }
                ]
            )
            return response['message']['content']
        
        except Exception as e:
            return f"生成回應時發生錯誤: {str(e)}"

# Flask Web應用
app = Flask(__name__)
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

@course_bot.handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_query = event.message.text
    response = course_bot.generate_response(user_query)
    
    try:
        course_bot.line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=response)
        )
    except LineBotApiError as e:
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
    app.run(port=5000)
