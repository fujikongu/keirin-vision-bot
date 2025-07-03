import os
import io
from google.cloud import vision
from PIL import Image
import openai

# OpenAIとGoogle認証の設定
openai.api_key = os.getenv("OPENAI_API_KEY")
google_credential_json = os.getenv("GOOGLE_CREDENTIAL_JSON")

if not google_credential_json:
    raise ValueError("GOOGLE_CREDENTIAL_JSON が未設定です")

# Google Cloudの認証ファイルを一時的に保存
with open("google_key.json", "w") as f:
    f.write(google_credential_json)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_key.json"

# === OCR処理 ===
def detect_text_from_image(image_path: str) -> str:
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if not texts:
        return ""
    return texts[0].description

# === ChatGPTで予想生成（車番のみ・3連単）===
def generate_keirin_prediction(ocr_text: str) -> str:
    prompt = f"""
以下は競輪の出走表（OCR結果）です。車番（数字）のみを使用し、選手名を一切含めないでください。
この情報をもとに、1着に来そうな選手3人、2着に来そうな選手4人、3着に来そうな選手7人を選び、
「3連単」の45点フォーメーションを車番（例：1→2→3）だけで出力してください。

出走表：
{ocr_text}

出力形式（例）：
1. 3→1→7
2. 3→1→5
3. 3→1→6
...
※選手名や「車番○○」などの表現は不要、数字のみ使用。
※「45点」などの点数表記は不要です。
※3連複ボックスは出力しないでください。
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1500
    )
    return response.choices[0].message['content']

# === 画像バイナリデータを処理して予想を返す関数 ===
def process_image_and_predict(image_data: bytes) -> str:
    temp_filename = "temp_image.jpg"
    with open(temp_filename, "wb") as f:
        f.write(image_data)
    
    ocr_result = detect_text_from_image(temp_filename)
    if not ocr_result.strip():
        return "画像から文字が読み取れませんでした。画像の鮮明さや解像度をご確認ください。"

    prediction = generate_keirin_prediction(ocr_result)
    return prediction
