from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage

import os
from vision_ocr import process_image_and_predict

app = Flask(__name__)

# 環境変数取得
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GOOGLE_CREDENTIAL_JSON = os.getenv("GOOGLE_CREDENTIAL_JSON")

# チェック
if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise ValueError("LINE環境変数未設定")
if not GOOGLE_CREDENTIAL_JSON:
    raise ValueError("Google認証キーが未設定")

# Cloud Vision API 認証ファイル出力
with open("google_key.json", "w") as f:
    f.write(GOOGLE_CREDENTIAL_JSON)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_key.json"

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

# === 画像受信処理 ===
@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    message_content = line_bot_api.get_message_content(event.message.id)
    
    # 🔽 ここが最重要：正しくバイトデータを取得する
    image_data = message_content.content

    try:
        result = process_image_and_predict(image_data)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=result))
    except Exception as

ありがとうございます。アップロードされた `main 2.py` を確認しました。  
このファイルが `"app"` を参照していたり、構成上の問題を抱えていたりするかを確認して、**vision_ocr.py と正しく連携した最小構成の `main.py`（ファイル数1）** を以下に **完全書き出し** します。

---

## ✅ あなたの目的に沿った `main.py`（完全版）

```python
import os
import json
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageMessage
)
from vision_ocr import process_image_and_predict

# Flaskアプリ
app = Flask(__name__)

# === 環境変数の取得 ===
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GOOGLE_CREDENTIAL_JSON = os.getenv("GOOGLE_CREDENTIAL_JSON")

# === チェック ===
if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise ValueError("LINEの環境変数が未設定です。")
if not GOOGLE_CREDENTIAL_JSON:
    raise ValueError("GOOGLE_CREDENTIAL_JSON 環境変数が未設定です。")

# === 認証ファイルの一時保存 ===
with open("google_key.json", "w") as f:
    f.write(GOOGLE_CREDENTIAL_JSON)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_key.json"

# === LINE初期化 ===
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# === Webhook受信エンドポイント ===
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

# === 画像受信時の処理 ===
@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    message_content = line_bot_api.get_message_content(event.message.id)
    image_data = b''.join(chunk for chunk in message_content.iter_content(chunk_size=1024))

    try:
        prediction = process_image_and_predict(image_data)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=prediction))
    except Exception as e:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"エラーが発生しました: {str(e)}"))

# === テキスト受信時の処理 ===
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    text = event.message.text
    if "テスト" in text:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="Botは正常に動作しています。出走表画像を送ってください。"))
    else:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="出走表の画像を送ってください。"))

# === アプリ起動 ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
