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

# === ChatGPTで予想生成 ===
def generate_keirin_prediction(ocr_text):
    prompt = f"""
以下は競輪の出走表です。この情報をもとに、1着→2着→3着の可能性が高い選手を予想し、3連単フォーメーションを45〜50点で出力してください。

出走表：
{ocr_text}

出力形式：
・予想は車番（例：1→2→3）のみで表記してください。選手名は不要です。
・出力点数は45〜50点以内にしてください。
・出走表の並び順（上にあるから強いなど）は一切考慮せず、
・選手の実力、競走得点、勝率などの要素に基づいて判断してください。
・特定の車番を偏って選ばないでください。
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
