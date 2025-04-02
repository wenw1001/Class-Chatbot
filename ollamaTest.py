import ollama
import time

# 設置 Ollama API 的模型
model = "deepseek-r1"  # 這裡可以選擇你想要測試的模型

# 發送請求來啟動模型
response = ollama.chat(model=model, messages=[
                    {
                        'role': 'system', 
                        'content': '用中文簡短回答問題。1+1等於多少'
                    },
                    {"role": "user", "content": "Hello, how are you?"}])

# 檢查返回的內容
if 'message' in response and 'content' in response['message']:
    content = response['message']['content']
    if content:
        print(content)
    else:
        print("No content returned.")
else:
    print("No message or content in the response.")
