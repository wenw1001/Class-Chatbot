import ollama
import time
import re

# 設定模型名稱
model = "llama2:13b-chat" # qwen:7b-chat, deepseek-r1, llama2:13b-chat

# 建立對話歷史（會話狀態）
system_messages = [
                    {"role": "system", "content": """你是「機器視覺課程」的助教機器人，僅提供課程公告、課堂大綱、與作業規範說明。你禁止提供任何形式的程式碼或邏輯內容，或課程無關的回答。

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
                    """}
                ]

assistant_messages = [{"role": "system", "content":"""
                        📌 目前公告內容如下：

                        作業一：
                        - 主題：灰階轉換與直方圖均衡化
                        - 說明：將彩色圖片轉為灰階後，實作直方圖均衡化以提升對比度
                        - 限制：不可使用 cv2.equalizeHist()，需自行實作演算法。可以使用cv2.imread()基本函式。
                        - 繳交期限：2025/03/08（五）23:59
                        - 提醒：需附上原始圖片、處理後圖片與簡要說明（PDF）

                        作業二：
                        - 主題：影像平移、旋轉與縮放
                        - 內容：根據給定的參數進行仿射變換，輸出前後對照圖
                        - 限制：禁止使用cv2.warpAffine()、cv2.getRotationMatrix2D()等現有功能函式。可以使用cv2.imread()基本函式。
                        - 繳交方式：命名格式 HW2_學號_姓名.zip，並上傳至 EEClass
                        - 繳交期限：2025/03/22（五）23:59

                        作業三：
                        - 主題：實作邊緣檢測功能
                        - 限使用 OpenCV 的基本操作（不可使用 `cv2.Canny()`）。可以使用cv2.imread()基本函式。
                        - 上傳期限：2025/04/19 23:59
                        - 上傳方式：至 eeclass 上傳程式壓縮檔與說明文件

                        課堂主題（第十週）：
                        - 邊緣偵測原理（Sobel, Prewitt）
                        - 二值化與形態學操作

                        課程公告：
                        期中考：2025/04/23（週三）上課時間進行，請攜帶計算機與學生證
                        小組分組提醒：請於 4/15 前完成期末專題小組分組（3～4人為限），逾期將由助教隨機分配
                        缺席補件：任何因故缺席者須於兩週內完成補交程序，並主動告知助教
                        學期末報告：題目不限，但需與機器視覺有實作關聯，報告日期為 6/19（三）
                    """}]

print("請輸入你的問題，若需要記憶功能，請輸入'memory mode'，將根據前三次的對話回答問題（輸入 exit 結束）：")

memory_mode = False
MAXIMUM_MEMORY = 3 # 最大記憶量

while True:
    user_input = input("你：")
    if user_input.strip().lower() == "exit":
        print("結束對話。")
        break

    if user_input.strip().lower() == "memory mode":
        memory_mode = True
        print("-- 記憶模式已開啟 --")
        print("請輸入你的問題（輸入 memory mode off 關閉記憶模式，或輸入 exit 結束）：")
        user_input = input("你：")
        history_messages = []
    
    if user_input.strip().lower() == "memory mode off":
        if memory_mode:
            print("-- 記憶模式已關閉 --")
            memory_mode = False
            continue
        else:
            print("記憶模式未開啟，請重新輸入你的問題：")
            continue

    if memory_mode:
        while len(history_messages)>MAXIMUM_MEMORY*2:
            history_messages.pop(0)
        print(f"(目前記憶量為前 {len(history_messages)//2} 次的對話)")
        # 將使用者輸入加入對話歷史
        history_messages.append({"role": "user", "content": user_input})
        prompt = system_messages + assistant_messages + history_messages
    else:
        prompt = system_messages+ assistant_messages + [{"role": "user", "content": user_input}]

    # 發送請求
    try:
        response = ollama.chat(model=model, messages=prompt)

        # 檢查是否有內容
        if 'message' in response and 'content' in response['message']:
            content = response['message']['content']
            content = re.sub(r'.*?</think>\n*', '', content, flags=re.DOTALL)
            print("AI：" + content)
            if memory_mode:
                # 將 AI 回覆也加入對話歷史
                history_messages.append({"role": "assistant", "content": content})
        else:
            print("AI 沒有回應。")

    except Exception as e:
        print(f"發送請求時發生錯誤：{e}")
