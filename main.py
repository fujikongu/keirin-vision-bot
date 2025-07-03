import os
import json
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, TextSendMessage

from vision_ocr import detect_text_from_image

# サービスアカウントのJSONを環境変数からファイルに保存
credential_json = os.environ.get("GOOGLE_CREDENTIAL_JSON")
with open("service_account.json", "w") as f:
    f.write(credential_json)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account.json"

# Flaskアプリ起動
app = Flask(__name__)

# LINE Bot設定
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')
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

# 画像メッセージ受信時の処理
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    # 一時的に画像を保存
    message_content = line_bot_api.get_message_content(event.message.id)
    image_path = f"/tmp/{event.message.id}.jpg"
    with open(image_path, "wb") as f:
        for chunk in message_content.iter_content():
            f.write(chunk)

    # OCRでテキスト抽出
    extracted_text = detect_text_from_image(image_path)

    # 抽出結果を返信
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=f"OCR解析結果:\n{extracted_text}")
    )

# アプリ起動（Render向け）
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
