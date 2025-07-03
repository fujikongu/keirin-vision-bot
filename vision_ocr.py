import os
import io
from google.cloud import vision
from PIL import Image
import openai

# === 環境変数から APIキー取得 ===
openai.api_key = os.getenv("OPENAI_API_KEY")
google_credential_json = os.getenv("GOOGLE_CREDENTIAL_JSON")

if not google_credential_json:
    raise ValueError("GOOGLE_CREDENTIAL_JSON 環境変数が設定されていません")

# Google Cloud 認証設定ファイルを一時的に保存
with open("google_key.json", "w") as f:
    f.write(google_credential_json)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_key.json"

# === OCR処理（画像バイトデータから直接処理）===
def detect_text_from_bytes(image_bytes):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if not texts:
        return ""
    return texts[0].description

# === ChatGPTで予想生成（出力修正版）===
def generate_keirin_prediction(ocr_text):
    prompt = f"""
以下は競輪の出走表です。この情報をもとに、1着→2着→3着の可能性が高い選手を予想し、3連単フォーメーションを80点以内で出力してください。

出走表：
{ocr_text}

出力形式：
- 有力選手7名（表形式）
- 1着候補4人、2着候補5人、3着候補7人で構成
- 3連単フォーメーションを的中率が高い順で出力（80点以内）
- 出力は「1→2→3」の形式（“車”は付けない）
- 3連複や点数表示は不要です（出力しないでください）
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1500
    )

    return response.choices[0].message['content']

# === 最終処理（バイナリ画像データ対応）===
def process_image_and_predict(image_bytes):
    ocr_result = detect_text_from_bytes(image_bytes)
    if not ocr_result:
        return "OCRで文字を読み取れませんでした。画像を確認してください。"
    prediction = generate_keirin_prediction(ocr_result)
    return prediction
