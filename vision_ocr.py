import os
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

# === プロンプト生成関数（逃げ偏り防止バージョン） ===
def generate_keirin_prompt(race_info: str) -> str:
    prompt = f"""
以下は競艇の出走表です。展示タイム、モーター2連対率、コース進入、スタート展示、F持ち、地元選手かどうかなどの情報から、展開を予想し、的中確率の高い3連単フォーメーションを出力してください。

{race_info}

【あなたの役割】
プロの競艇予想AIとして、艇ごとの脚質・進入コース・展示タイムを重視し、展開を読み取ってください。

【展開予想の指針】
- インが強い競艇の基本傾向を考慮（1コースが有利）
- モーター2連対率が高く展示タイムも良い選手は加点
- スタート展示の良さ・スタート順（ST）を評価
- 枠なり進入でない場合はスタート隊形と仕掛けを予測
- 地元選手や伸び足がある選手は展開に応じて加点
- F持ちやスタート遅れは減点材料

【評価の基本】
- 1着候補はインからの順に強力な選手を評価（最大3艇）
- 2着候補には展開次第で浮上するまくり差し艇も含める
- 3着候補には残りの全艇から展開に応じて選出

【出力形式】
1着候補: x,x,x  
2着候補: x,x,x,x,x  
3着候補: x,x,x,x,x,x  

フォーメーション:  
x→x→x  
x→x→x  
...（最大20点まで）

※出力は最大で20点以内にしてください  
※艇番号のみで構成し、選手名や解説文は不要  
※1着候補にいない艇が1着になる組み合わせは出力しないでください  
※的中確率が高い順に並べてください
"""
    return prompt.strip()

# === ChatGPTによる展開付き3連単予想生成 ===
def generate_keirin_prediction_with_race_scenario(ocr_text):
    prompt = generate_keirin_prompt(ocr_text)
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1500
    )
    return response.choices[0].message['content']

# === 画像データ（bytes）を受け取って予想出力 ===
def process_image_and_predict(image_bytes):
    ocr_result = detect_text_from_image_bytes(image_bytes)
    if not ocr_result:
        return "OCRで文字を読み取れませんでした。画像を確認してください。"
    prediction = generate_keirin_prediction_with_race_scenario(ocr_result)
    return prediction
