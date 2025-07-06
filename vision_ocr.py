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
以下は競輪の出走表です。各選手の情報（競走得点、勝率、連対率、決まり手、B/H/S、直近成績、地区など）をもとに、展開を読み取り、的中確率の高い3連単フォーメーションを出力してください。

{race_info}

【あなたの役割】
プロの競輪予想AIとして、展開を重視し予想を構築してください。

【展開予想】
- スタートで前を取る選手
- 主導権を取る可能性のあるライン
- まくり・差しで勝負する選手
- 番手から抜け出しそうな選手

【評価基準（全て反映）】
- 展開を最重視し、勝率・連対率・得点は補助的に考慮
- 脚質（逃げ・捲り・差し）やB/H/Sから展開を予測
- 差し脚を活かせる番手選手を1着候補に優先して選出
- 逃げ選手は展開次第で1〜2着候補に含めるが、3人とも逃げになるような偏った予想は避ける
- まくり・自在型は2着候補、追い込み型は3着候補に評価
- 直近成績が良い選手は加点
- 同じラインから3人選出しないこと（バランスよく構成）

【特記事項】
このレースは「3分戦（3つのライン構成）」です。

【展開を読む際の注意点】
- 主導権を取りそうなラインとその番手を重視
- まくりのタイミングや差しが決まりやすい展開を考慮
- 風やバンク特性に応じた差し・まくり有利の傾向を意識
- 同地区の並び、ラインの先頭が潰れる場合の展開も想定

【新人・若手選手の評価について】
- 期別が120期以降の新人であっても、勝率70%以上、連対率・3連対率が高い場合は実力があると判断する。
- 得点が低くても、成績が極端に良い新人選手は1着2着3着候補に含めてよい。
- ただし他の選手との展開や脚質、ライン構成のバランスを考慮すること。

【出力形式】
1着候補: x,x,x  
2着候補: x,x,x,x,x  
3着候補: x,x,x,x,x,x  

フォーメーション:  
x→x→x  
x→x→x  
...（最大50点まで）

※選手名・解説文は出力せず、車番の数字のみで構成すること  
※1着候補にいない選手が1着になる組み合わせは出力しないこと  
※的中確率が高い順に並べること
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
