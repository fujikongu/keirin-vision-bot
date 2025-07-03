import os
import io
from google.cloud import vision
from PIL import Image
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

# Google Cloud 認証設定ファイルを環境変数から取得
google_credential_json = os.getenv("GOOGLE_CREDENTIAL_JSON")
if not google_credential_json:
    raise ValueError("GOOGLE_CREDENTIAL_JSON が未設定です")

with open("google_key.json", "w") as f:
    f.write(google_credential_json)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_key.json"

# === OCR処理 ===
def detect_text_from_image_bytes(image_bytes):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description if texts else ""

# === ChatGPT で予想を生成 ===
def generate_prediction(ocr_text):
    prompt = f"""
以下は競輪の出走表です。この出走表から次の条件に従って予想してください。

【条件】
- 1着に来そうな選手を3人
- 2着に来そうな選手を5人
- 3着に来そうな選手を6人
- 組み合わせは「1→2→3」のように、車番のみで出力する
- 名前は一切出さない
- 予想点数は45〜50点に自動調整
- 3連複の記述は一切不要
- 補足文や注釈も出さず、予想の組み合わせのみを羅列する

出走表：
{ocr_text}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1500
    )

    return response.choices[0].message["content"]

# === メイン処理 ===
def process_image_and_predict(image_bytes):
    ocr_text = detect_text_from_image_bytes(image_bytes)
    if not ocr_text:
        return "OCRで文字を読み取れませんでした。画像を確認してください。"
    return generate_prediction(ocr_text)
