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

# === OCR処理 ===
def detect_text_from_image(image_path):
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if not texts:
        return ""
    return texts[0].description

# === ChatGPTで予想生成（プロンプト修正済み） ===
def generate_keirin_prediction(ocr_text):
    prompt = f"""
以下は競輪の出走表です。この情報をもとに、1着→2着→3着の可能性が高い選手を予想し、3連単フォーメーションを出力してください。

【条件】
- 選手名を使わず「車番（1〜9）」のみを使い、「1 → 2 → 3」のように出力してください（「車」の文字は使わないでください）
- 有力選手を7名選出（表形式）
- 1着候補3人、2着候補4人、3着候補7人で構成してください
- 出力は3連単フォーメーションのみ（順不同で最大45点）にしてください
- 「3連複ボックス」や点数表示は一切不要です
- 出力は日本語でお願いします

出走表（OCR結果）：
{ocr_text}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1500
    )

    return response.choices[0].message['content']

# === 画像ファイルを処理して最終予想を返す ===
def process_image_and_predict(image_path):
    ocr_result = detect_text_from_image(image_path)
    if not ocr_result:
        return "OCRで文字を読み取れませんでした。画像を確認してください。"
    prediction = generate_keirin_prediction(ocr_result)
    return prediction
