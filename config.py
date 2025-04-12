import os
from dotenv import load_dotenv

# 清除先前可能設定的環境變數
os.environ.pop('OLLAMA_MODEL', None)

# 載入 .env 文件
load_dotenv("./.env")

# Line Bot 配置
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', '')
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET', '')

# Ollama 模型配置
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama2:13b-chat')
# print("(config.py) 目前使用的模型是：", OLLAMA_MODEL)
# Webhook 配置
WEBHOOK_URL = os.getenv('WEBHOOK_URL', 'http://localhost:5000/webhook')

# 額外的安全檢查
def validate_config():
    """驗證重要配置是否已正確設置"""
    errors = []
    
    if not LINE_CHANNEL_ACCESS_TOKEN:
        errors.append("未設置 LINE_CHANNEL_ACCESS_TOKEN")
    
    if not LINE_CHANNEL_SECRET:
        errors.append("未設置 LINE_CHANNEL_SECRET")
    
    if errors:
        raise ValueError("\n".join(errors))

# 在導入時立即進行配置驗證
validate_config()
