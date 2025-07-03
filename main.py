import import os
import json
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageMessage
)
from vision_ocr import process_image_and_predict

app = Flask(__name__)

# === 環境変数の読み込み ===
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GOOGLE_CREDENTIAL_JSON = os.getenv("GOOGLE_CREDENTIAL_JSON")

# === 環境変数チェック ===
if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise ValueError("LINEの環境変数が未設定です。")
if not GOOGLE_CREDENTIAL_JSON:
    raise ValueError("GOOGLE_CREDENTIAL_JSON 環境変数が設定されていません。")

# === Cloud Vision API 認証用に環境変数を設定 ===
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIAL_JSON

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# === ルート設定 ===
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

# === メッセージイベント処理 ===
@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    message_content = line_bot_api.get_message_content(event.message.id)
    image_data = b''.join(chunk for chunk in message_content.iter_content(chunk_size=1024))
    
    try:
        prediction_result = process_image_and_predict(image_data)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=prediction_result))
    except Exception as e:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"エラーが発生しました: {str(e)}"))

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    text = event.message.text
    if "テスト" in text:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="Botは正常に動作しています。出走表画像を送ってください。"))
    else:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="画像を送信してください（出走表）"))

# === Flask 起動 ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
