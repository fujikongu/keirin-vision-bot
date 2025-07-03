import os
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, ImageMessage, TextSendMessage
from linebot.exceptions import InvalidSignatureError
from google.cloud import vision
import io
import openai
from itertools import product

app = Flask(__name__)

# 環境変数（Render で設定）
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GOOGLE_CREDENTIAL_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
openai.api_key = OPENAI_API_KEY

# Vision APIクライアント
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIAL_JSON
vision_client = vision.ImageAnnotatorClient()


@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"


@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    # 画像を取得
    message_content = line_bot_api.get_message_content(event.message.id)
    image_data = b""
    for chunk in message_content.iter_content():
        image_data += chunk

    # Vision APIでOCR
    image = vision.Image(content=image_data)
    response = vision_client.text_detection(image=image)
    texts = response.text_annotations
    if not texts:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="画像からテキストを読み取れませんでした。"))
        return

    full_text = texts[0].description

    # ChatGPTに選手の評価と3連単予測依頼（車番のみで）
    system_prompt = "あなたは競輪予想の専門家です。以下の出走表から有力な選手を車番のみで予想し、1着候補3人・2着候補4人・3着候補7人を選び、重複なし3連単45点（車番形式）を出力してください。選手名や勝率も考慮して構いませんが、出力には車番しか使わないでください。"

    user_prompt = f"出走表の内容:\n{full_text}\n\n3連単予想を出力してください（例：3→1→7）"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    prediction_text = response["choices"][0]["message"]["content"]
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=prediction_text))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
