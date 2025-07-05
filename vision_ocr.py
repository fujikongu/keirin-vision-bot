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

# === ChatGPTによる展開予想付き3連単予想生成（強化プロンプト） ===
def generate_keirin_prediction_with_race_scenario(ocr_text):
    prompt = f"""
以下は競輪の出走表です。

各選手について、競走得点、勝率、2連対率、3連対率、決まり手（逃げ・捲り・差し）、バック本数（B）、捲られた回数（H）、スタート数（S）、直近成績、地区などが記載されています。

【あなたの役割】
プロの競輪予想AIとして、展開を踏まえて、的中確率の高い3連単フォーメーションを出力してください。

【展開予想】
以下の展開を最初に考慮し、その上で予想してください：
- どの選手がスタートで前を取りそうか
- どのラインが主導権（先行）を取る可能性が高いか
- 誰がまくり・差しで勝負する展開になるか
- 番手から抜け出しそうな選手がいるか

【評価基準（すべて反映してください）】
- 競走得点が高い選手を基本に評価
- 勝率・連対率が高い選手を優先
- 脚質（逃げ・捲り・差し）とB/H/Sを参考に展開を予測
- 近況成績や直近着順の良い選手を重視
- 同地区やラインの可能性を加味（同じ地区が並ぶと有利）
- 先行力のある選手がいないレースでは、差し・捲りが有利
- 強力な先行選手がいる場合は、そのラインの番手選手を高評価
- 特定の車番に偏らず、バランスよく候補を選出

【出力ルール（絶対厳守）】
- 1着候補：3人（この3人以外を1着にしてはいけません）
- 2着候補：5人（1着候補を含んでもよい）
- 3着候補：6人（1着・2着候補を含んでもよい）
- 出力はこの候補のみで構成すること。候補外の車番は使用禁止。
- 出力は車番のみ（例：1→2→3）で、解説文・選手名は不要
- 出力フォーメーションは的中確率が高い順に45〜50点出力すること

【出走表】
{ocr_text}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1500
    )
    return response.choices[0].message['content']


# === 出力バリデーション：1着候補外が含まれていないか検査し除外 ===
def validate_predictions(predictions_text, ichaku_candidates):
    valid_lines = []
    for line in predictions_text.splitlines():
        if "→" in line:
            first = line.split("→")[0].strip()
            if first in ichaku_candidates:
                valid_lines.append(line.strip())
    return "\n".join(valid_lines)


# === メイン処理：画像から3連単予想を生成 ===
def process_image_and_predict(image_bytes):
    ocr_result = detect_text_from_image_bytes(image_bytes)
    if not ocr_result:
        return "OCRで文字を読み取れませんでした。画像を確認してください。"

    raw_prediction = generate_keirin_prediction_with_race_scenario(ocr_result)

    # ★ ChatGPT出力から1着候補を抽出
    ichaku_candidates = []
    for line in raw_prediction.splitlines():
        if "1着候補" in line:
            ichaku_candidates = [x.strip() for x in line.split(":")[1].split(",")]
            break

    # ★ 出力のバリデーション処理
    filtered_prediction = validate_predictions(raw_prediction, ichaku_candidates)

    return filtered_prediction if filtered_prediction else "有効な3連単予想が取得できませんでした。"
