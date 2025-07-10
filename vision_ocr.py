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
is_women_only = "B1" in race_info and all(name in race_info for name in ["さん", "子", "花", "美", "夏"])  # 簡易判定

    base_prompt = f"""以下は競艇（ボートレース）の出走表情報です。展示タイムは考慮せず、**モーター性能・コース・スタート傾向・選手の脚質・近況成績**などから展開を予測し、**的中確率の高い3連単フォーメーション**を構築してください。ん
    
    if is_women_only:
        base_prompt += """
以下は競艇（ボートレース）の出走表情報です。展示タイムは考慮せず、**モーター性能・コース・スタート傾向・選手の脚質・近況成績**などから展開を予測し、**的中確率の高い3連単フォーメーション**を構築してください。

{race_info}

【あなたの役割】
プロの競艇予想AIとして、**展開を軸にした精度の高い3連単予想**を行ってください。

【展開予測で重視すべきポイント】
- 1コースの信頼度（イン逃げの成功率）
- モーター2連対率（出足・伸び・安定感）
- 平均ST（スタートタイミング）とF（フライング）歴の有無
- 脚質（まくり、差し、自在型）のタイプとコースとの相性
- 地元選手かどうか（地の利、乗り慣れた水面）
- 枠番の利を活かせるか（特に1～3号艇）
- 外枠（4～6号艇）選手の攻め・展開に乗る力
- ピット離れが早く進入に変化を起こす可能性

【追加の判断基準】
- スタート巧者が複数いるときはまくり展開の可能性を重視
- 複数F持ちがいるとスタート全体が慎重になり、差し展開が有利になる
- モーター性能が抜けて高い選手は枠不利でも加点対象
- B級選手でもスタートが良く脚もある場合は展開で浮上可能と判断
- 1コースがF持ちまたはST遅い場合は2～4コースの攻めを重視すること

【展開に恵まれそうな選手の評価】
- 前づけやスロー進入によって展開の恩恵を受けやすい選手
- 先攻選手の後ろで差しに構えられる自在型
- 攻める選手が作る展開に乗じて浮上しそうな差し屋や2番差し
- 枠なりでも隣がF持ちやST不安定でチャンスが広がる艇

これらに該当する場合、たとえ地味な成績でも3着候補や穴として評価対象に含めてください。
""".strip()

    if is_women_only:
        base_prompt += """

【女子戦における特有の注意点】
- 一般的にスタートが慎重になりやすく、差し展開が増えやすい傾向があります。
- モーター性能の影響が結果に直結しやすく、2連対率は信頼度が高い材料です。
- 展開に乗りやすい選手（差し屋・自在型）が浮上しやすいので評価を忘れずに。
"""

    base_prompt += """

【出力形式】
1着候補: x,x,x  
2着候補: x,x,x,x  
3着候補: x,x,x,x,x  

フォーメーション:  
x→x→x  
x→x→x  
...（最大20点まで）

※すべての出力は数字（艇番）のみで構成してください  
※解説・コメント文は一切出力しないこと  
※1着候補にいない艇が1着になる組み合わせは出力しない  
※1着候補に含まれていない選手が1着になる組み合わせは出力しないこと  
※展開に整合性があるように構成すること（例：捲り展開に差し艇が1着は矛盾）  
※展開の恩恵がありそうな選手も着候補として適切に評価すること  
※的中確率が高い順に並べること
""".rstrip()

    return base_prompt

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
