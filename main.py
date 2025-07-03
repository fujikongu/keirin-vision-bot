import os
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage

from vision_ocr import process_image_and_predict

app = Flask(__name__)

# 環境変数取得
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GOOGLE_CREDENTIAL_JSON = os.getenv("GOOGLE_CREDENTIAL_JSON")

# チェック
if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise ValueError("LINE環境変数が未設定です")
if not GOOGLE_CREDENTIAL_JSON:
    raise ValueError("Google認証キーが未設定です")

# Google Cloud 認証ファイル保存
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

# 画像受信時の処理
@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    message_content = line_bot_api.get_message_content(event.message.id)
    image_data = b''.join(chunk for chunk in message_content.iter_content(1024))

    try:
        result = process_image_and_predict(image_data)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=result))
    except Exception as e:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"エラーが発生しました: {str(e)}"))

# テキスト受信時の処理
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    if event.message.text == "テスト":
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="Botは正常です。画像を送ってください。"))
    else:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="出走表の画像を送ってください。"))

# アプリ起動
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
