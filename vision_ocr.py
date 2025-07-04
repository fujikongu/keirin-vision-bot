
import os
import io
from google.cloud import vision
import openai

# === 環境変数から API キーを取得 ===
openai.api_key = os.getenv("OPENAI_API_KEY")
google_credential_json = os.getenv("GOOGLE_CREDENTIAL_JSON")

if not google_credential_json:
    raise ValueError("GOOGLE_CREDENTIAL_JSON 環境変数が設定されていません")

# === Google Cloud 認証ファイルを一時保存 ===
with open("google_key.json", "w") as f:
    f.write(google_credential_json)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_key.json"

# === OCR処理：画像データ（bytes）から文字認識 ===
def detect_text_from_image_bytes(image_bytes):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if not texts:
        return ""
    return texts[0].description

# === ChatGPTによる予想生成 ===
def generate_keirin_prediction(ocr_text):
    prompt = f"""
以下は競輪の出走表です。

各選手について、競走得点、勝率、2連対率、3連対率、決まり手（逃げ・捲り・差し）、バック本数（B）、捲られた回数（H）、スタート数（S）、直近成績、地区名などが記載されています。

【あなたの役割】
あなたはプロの競輪予想AIです。以下の評価基準すべてをふまえ、展開を予測し、的中確率の高い3連単フォーメーションを構成してください。

【評価基準】
- 競走得点が高い選手を基本に評価
- 勝率・連対率が高い選手を優先
- 近況成績や直近着順の良い選手を重視
- 同地区やラインの可能性を加味（同じ地区が並ぶと有利）
- 先行力のある選手がいないレースでは、差し・捲りが有利
- 特に強力な先行選手がいる場合は、その番手選手を高評価
- 脚質（逃げ・捲り・差し）とB/H/Sを参考に展開を予測
- 特定の車番に偏らず、バランスよく候補を選出
- 表の並び順や上位配置に引きずられないよう注意

【出力ルール】
- 1着候補：3人
- 2着候補：5人
- 3着候補：6人
- 上記を使った3連単フォーメーションを **45～50点以内** で構成
- 出力は **車番のみ**（例：1→2→3）とし、選手名・解説文などは出力しない

【出走表】
{ocr_text}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1500
    )
    return response.choices[0].message['content']

# === 画像データ（bytes）を受け取って予想出力 ===
def process_image_and_predict(image_bytes):
    ocr_result = detect_text_from_image_bytes(image_bytes)
    if not ocr_result:
        return "OCRで文字を読み取れませんでした。画像を確認してください。"
    prediction = generate_keirin_prediction(ocr_result)
    return prediction
