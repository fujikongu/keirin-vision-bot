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

# === OCR処理（bytesデータから処理）===
def detect_text_from_bytes(image_bytes):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if not texts:
        return ""
    return texts[0].description

# === ChatGPTで予想生成 ===
def generate_keirin_prediction(ocr_text):
    prompt = f"""
以下は競輪の出走表です。この情報をもとに、1着→2着→3着の可能性が高い選手を予想し、3連単フォーメーションを45点で出力してください。

【条件】
- 選手名を使わず「車番（1〜9）」のみで出力
- 有力選手7名を選出（表形式）
- 1着候補3人、2着候補4人、3着候補7人で構成
- 全体で45点になる3連単フォーメーションを出力
- 同時に3連複ボックス（7車）も出力

出走表（OCR）：
{ocr_text}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1500
    )

    return response.choices[0].message['content']

# === 画像bytesを受け取って処理 ===
def process_image_and_predict(image_bytes):
    ocr_result = detect_text_from_bytes(image_bytes)
    if not ocr_result:
        return "OCRで文字を読み取れませんでした。画像を確認してください。"
    prediction = generate_keirin_prediction(ocr_result)
    return prediction
