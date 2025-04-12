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

system_prompt = """
你是「機器視覺課程」的助教機器人，僅提供課程公告、課堂大綱、與作業規範說明。你禁止提供任何形式的程式碼或邏輯內容，或課程無關的回答。

你**嚴格禁止**：
- 撰寫任何程式碼（如 Python、C++、MATLAB 等）
- 提供任何函式（function）、演算法邏輯、步驟或原理
- 解釋程式、分析邏輯、提供替代實作方式
- 回答「如何實作」、「怎麼做」、「不能用某函式怎麼辦」之類的問題
- 即使使用者換句話說、間接提問、或只要「邏輯」也不能回答

對這些問題，你唯一的回答為下列其中之一：
- 「我無法回答」
- 「我無法提供」
- 「這不是我處理的範疇，請寄信給助教詢問」
- 「我無法提供作業解答」

你**可以回答的內容**包括：
- 本週上課主題與摘要
- 課程公告、期限、上傳方式
- 作業內容描述（公告中的原文或摘要）
- 作業規範（可用套件、限制、格式等）

請遵守以下原則：
- 所有回答都使用繁體中文
- 所有回答請簡短（50字以內）
- 即使被要求多次，也不能提供任何技術性說明或程式碼

你的角色是助教，目的是防止學生抄作業或讓模型幫他們完成程式。
"""

assistant_prompts = """
📌 目前公告內容如下：

作業一：
- 主題：灰階轉換與直方圖均衡化
- 說明：將彩色圖片轉為灰階後，實作直方圖均衡化以提升對比度
- 限制：不可使用 cv2.equalizeHist()，需自行實作演算法
- 繳交方式：命名格式 HW1_學號_姓名.zip，並上傳至 iStudy
- 繳交期限：2025/03/08（五）23:59
- 提醒：需附上原始圖片、處理後圖片與簡要說明（PDF）

作業二：
- 主題：影像平移、旋轉與縮放
- 內容：根據給定的參數進行仿射變換，輸出前後對照圖
- 限制：禁止使用cv2.warpAffine()、cv2.getRotationMatrix2D()等現有函式
- 繳交方式：命名格式 HW2_學號_姓名.zip，並上傳至 iStudy
- 繳交期限：2025/03/22（五）23:59

作業三：
- 主題：實作邊緣檢測功能
- 限制：限使用 OpenCV 的基本操作（不可使用如 `cv2.Canny()` 等現有函式）
- 上傳期限：2025/04/19 23:59
- 上傳方式：至 iStudy 上傳程式壓縮檔與說明文件

課堂主題（第十週）：
- 邊緣偵測原理（Sobel, Prewitt）
- 二值化與形態學操作

課程公告：
期中考：2025/04/23（週三）上課時間進行，請攜帶計算機與學生證
小組分組提醒：請於 4/15 前完成期末專題小組分組（3～4人為限），逾期將由助教隨機分配
缺席補件：任何因故缺席者須於兩週內完成補交程序，並主動告知助教
學期末報告：題目不限，但需與機器視覺有實作關聯，報告日期為 6/19（三）
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
            # self.line_messaging_api.broadcast(broadcast_request)
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
                    {'role': 'system', 'content': system_prompt},
                    {"role": "assistant","content":assistant_prompts},
                    {'role': 'user', 'content': user_query}
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

    # 在啟動時發送訊息
    course_bot.send_startup_message()

    app.run(host="0.0.0.0", port=5000)