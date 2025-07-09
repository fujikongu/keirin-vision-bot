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
def generate_keirin_prompt(race_info: str, keibajo: str = "") -> str:
    KEIBA_JO_FEATURES = {
        "東京": "直線が長く差しが決まりやすい。逃げ切りは難しい傾向。",
        "中山": "急坂があるため持久力勝負になりやすい。先行有利。",
        "京都": "コーナーが緩やかで末脚が活きる。差し・追い込みも届く。",
        "阪神": "直線の坂がタフ。先行〜差しが安定しやすい。",
        "中京": "ペースが速くなりやすく差しが有利な傾向。",
        "小倉": "小回りで逃げ・先行が有利。開幕週は特に前有利。",
        "札幌": "時計がかかりやすく、スタミナと先行力が必要。",
        "函館": "洋芝でパワー型向き。逃げ先行が残る。",
        "新潟": "直線が長く、末脚が問われる。追い込みも届く。",
        "福島": "小回りでコーナー４回。先行馬有利。",
        "中山ダート": "パワーと先行力が重要。外枠はやや不利。",
        "東京ダート": "差しが決まりやすいが、1600は外枠不利。",
        "地方競馬場": "基本的に小回りで先行有利。重馬場にも注意。",
    }
    feature = KEIBA_JO_FEATURES.get(keibajo, "特筆すべき傾向はありません。")

    prompt = f"""
以下は競馬の出走表です。出走馬の情報（枠順、脚質、成績、持ち時計など）をもとに、展開を予測し、的中確率の高い3連単フォーメーションを出力してください。

開催競馬場: {keibajo}
【競馬場の特徴】
{feature}

【あなたの役割】
プロの競馬予想AIとして、競馬場の特性や展開を考慮し、精度の高い3連単予想を構築してください。

【評価基準】
- 各馬の枠順、脚質、ペース予測から展開を読み取ること
- 逃げ・先行が有利な馬場ではその脚質を高評価
- 差し・追い込みが決まりやすい馬場では末脚重視
- 近走成績やクラス実績を考慮して総合的に評価
- 本命馬（1着候補）は信頼度重視で3頭以内に絞る
- 展開的に有利な2〜3着候補をフォローして構成

【出力形式】
1着候補: x,x,x  
2着候補: x,x,x,x,x  
3着候補: x,x,x,x,x,x  

フォーメーション:  
x→x→x  
x→x→x  
...（最大20点まで）

※馬番のみで出力してください（解説不要）  
※1着候補にいない馬が1着になる組み合わせは出力しないでください  
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
