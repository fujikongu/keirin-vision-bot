import os
import io
from google.cloud import vision
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
def detect_text_from_bytes(image_bytes):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if not texts:
        return ""
    return texts[0].description

# === ChatGPTで車番予想のみ生成 ===
def generate_keirin_prediction(ocr_text):
    prompt = f"""
以下は競輪の出走表です。この情報をもとに、1着に来そうな選手3人、2着に来そうな選手5人、3着に来そうな選手6人を車番のみで予想し、それらの組み合わせから3連単の組み合わせを45〜50点以内で出力してください。
※選手名は一切表示せず、車番（数字）のみを使用してください。
※出力形式は以下に厳密に従ってください：

出力例：
1→2→3  
1→2→4  
...  
（合計〇点）

出走表：
{ocr_text}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1500
    )

    return response.choices[0].message['content']

# === 最終処理関数 ===
def process_image_and_predict(image_bytes):
    ocr_result = detect_text_from_bytes(image_bytes)
    if not ocr_result:
        return "出走表の文字を読み取れませんでした。画像を確認してください。"
    prediction = generate_keirin_prediction(ocr_result)
    return prediction
