import os
import io
from google.cloud import vision
from PIL import Image
import openai
import tempfile

# === 環境変数から API キー取得 ===
openai.api_key = os.getenv("OPENAI_API_KEY")
google_credential_json = os.getenv("GOOGLE_CREDENTIAL_JSON")

if not google_credential_json:
    raise ValueError("GOOGLE_CREDENTIAL_JSON 環境変数が設定されていません")

# Google Cloud 認証設定ファイルを一時的に保存
with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as temp_json:
    temp_json.write(google_credential_json)
    credential_path = temp_json.name

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path

# === OCR処理 ===
def detect_text_from_image_bytes(image_bytes):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if not texts:
        return ""
    return texts[0].description

# === ChatGPT で予想生成（車番のみ・45〜50点） ===
def generate_keirin_prediction(ocr_text):
    prompt = (
        "以下は競輪の出走表です。この情報をもとに、"
        "1着→2着→3着の可能性が高い選手を予想し、"
        "3連単フォーメーションを「車番のみ」で45〜50点で出力してください。\n\n"
        "条件：\n"
        "- 出力形式は車番のみ（例：1→2→3）\n"
        "- 1着候補：3人、2着候補：5人、3着候補：6人\n"
        "- 出力は最大50点まで、番号だけを羅列（解説不要）\n"
        "- 選手名の記載は禁止。車番だけを使うこと\n\n"
        f"出走表：\n{ocr_text}\n\n出力："
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1200
    )

    return response.choices[0].message["content"].strip()

# === LINE Bot から呼び出す関数 ===
def process_image_and_predict(image_data):
    try:
        ocr_text = detect_text_from_image_bytes(image_data)
        if not ocr_text.strip():
            return "出走表の文字を読み取れませんでした。画像を確認してください。"
        return generate_keirin_prediction(ocr_text)
    except Exception as e:
        return f"処理中にエラーが発生しました: {str(e)}"
