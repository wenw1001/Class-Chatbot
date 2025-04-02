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
from linebot.v3.messaging.models import TextMessage, PushMessageRequest, BroadcastRequest

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
    print(f"now:{datetime.now()}")
    return tw_time.strftime("%Y-%m-%d %H:%M:%S")

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
                        'content': '用中文簡短回答以下問題:' + user_query#'你是機器視覺課程的助教機器人，專門回答學生的課程相關問題。用繁體中文簡短回覆。'
                    },
                    {
                        'role': 'user', 
                        'content': user_query
                    }
                ]
            )
            
            response = re.sub(r'.*?</think>\n*', '', response['message']['content'], flags=re.DOTALL)
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
    print(f"{user_id} | 傳送訊息: {user_query}")
    response = course_bot.generate_response(user_query)
    print(f"機器人回覆: {response}")
    
    # replyToken=event.reply_token
    # messages=TextMessage(text=response)
    try:
        course_bot.line_messaging_api.reply_message(
            replyToken=event.reply_token,
            messages=[{'type': 'text', 'text': str(response)}]
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

    app.run(host="0.0.0.0", port=5000)