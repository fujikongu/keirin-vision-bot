import os
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, TextMessage, TextSendMessage
from vision_ocr import process_image_and_predict

# 環境変数の取得
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GOOGLE_CREDENTIAL_JSON = os.getenv("GOOGLE_CREDENTIAL_JSON")

# 検証
if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise ValueError("LINEの環境変数が未設定です。")
if not GOOGLE_CREDENTIAL_JSON:
    raise ValueError("GOOGLE_CREDENTIAL_JSON 環境変数が設定されていません。")

# Flask アプリ
app = Flask(__name__)
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

# 画像メッセージ受信時
@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    try:
        message_id = event.message.id
        image_content = line_bot_api.get_message_content(message_id)
        image_path = f"/tmp/{message_id}.jpg"

        # 画像を保存
        with open(image_path, "wb") as f:
            for chunk in image_content.iter_content():
                f.write(chunk)

        # 画像から予想を生成
        prediction_result = process_image_and_predict(image_path)

        # 結果を返信
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=prediction_result)
        )

    except Exception as e:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=f"エラーが発生しました：{str(e)}")
        )

# テキストメッセージ受信時（任意）
@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    if event.message.text.lower() == "使い方":
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="出走表の画像を送信すると、AIが3連単予想を返信します。")
        )
    else:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="画像を送信してください。")
        )

# アプリ起動
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
