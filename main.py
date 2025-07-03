import os
import json
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from vision_ocr import detect_text_from_image

# Flaskアプリ初期化
app = Flask(__name__)

# 環境変数からLINEの各種キーを取得
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GOOGLE_CREDENTIAL_JSON = os.getenv("GOOGLE_CREDENTIAL_JSON")

# JSON文字列をファイルに保存
if GOOGLE_CREDENTIAL_JSON:
    with open("service_account.json", "w") as f:
        f.write(GOOGLE_CREDENTIAL_JSON)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account.json"
else:
    raise ValueError("GOOGLE_CREDENTIAL_JSON 環境変数が設定されていません")

# LINE API初期化
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# Webhookエンドポイント
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"

# 画像メッセージを処理
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    if event.message.text.lower() == "ocr":
        reply_text = "画像を送ってください。"
    else:
        reply_text = f"「{event.message.text}」というメッセージを受信しました。"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
