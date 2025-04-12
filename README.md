# 機器視覺課程助教機器人 (LINE Chat Assistant)

本專案是一個 **LINE 聊天機器人**，專為「機器視覺課程」設計，提供學生課程相關資訊，如公告、作業規定、課堂主題等，同時 **嚴格禁止** 使用者請求程式碼、邏輯解釋或作業實作協助。

### 功能簡介

#### 支援的功能：
- 查詢課程公告
- 顯示本週課程內容
- 說明作業規定（使用限制、繳交方式等）
- 提醒系統公告與重要通知

#### 禁止的內容：
- **撰寫任何程式碼（Python、C++、MATLAB 等）**
- **提供任何函式、演算法邏輯、原理解釋**
- **協助解題、debug、程式分析**
- **回答與本課程無關的問題**

> 所有違規提問將收到固定拒答句，例如「我無法回答」、「這不是我處理的範疇」等。

---

### 系統架構
- LINE Messaging API
- LLM 模型（支援 Ollama：如 Qwen、LLaMA2）

---


## How to run?
1. 開啟ngrok伺服器，參考 [官方文件](https://dashboard.ngrok.com/get-started/setup/windows)
2. 在 LINE Developer Console 的 Messaging API 設定頁面，將 Webhook URL 設為：`https://你的ngrok網址/你的webhook路徑`
3. `pip install -r requirements.txt`，安裝相關套件
4. 執行 `machine-vision-chatbot.py`
即可成功開啟課程機器人

## 檔案說明
#### `promptTesting.py`
用來測試你的 prompt 是否正確遵守指定指令。

- 可自訂 `system_prompt` 與 `assistant_prompts`
- 可修改 `test_prompts` 來測試不同問題

#### `ollamaTest.py`
用來測試本地端的 Ollama 模型是否正常運作。

#### `machine-vision-chatbot.py`
啟動機器視覺課程專用的聊天機器人。

#### `requirements.txt`
列出執行上述腳本所需的所有套件。

## 參考
[ngrok](https://dashboard.ngrok.com/get-started/setup/windows)
[LINE Message API](https://developers.line.biz/zh-hant/services/messaging-api/)