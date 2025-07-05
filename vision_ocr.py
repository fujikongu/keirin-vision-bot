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

# === ChatGPTによる展開付き3連単予想生成 ===
def generate_keirin_prediction_with_race_scenario(ocr_text):
    prompt = f"""
以下は競輪の出走表です。

各選手について、競走得点、勝率、2連対率、3連対率、決まり手（逃げ・捲り・差し）、バック本数（B）、捲られた回数（H）、スタート数（S）、直近成績、地区などが記載されています。

【あなたの役割】
プロの競輪予想AIとして、展開を踏まえて、的中確率の高い3連単フォーメーションを出力してください。

【展開予想】
まず以下のポイントを読み取り、出走表の内容から展開を予測した上で3連単候補を構築してください：
- どの選手がスタートで前を取りそうか
- どのラインが主導権（先行）を取る可能性が高いか
- 誰がまくり・差しで勝負する展開になるか
- 番手から抜け出しそうな選手がいるか

【評価基準（すべて反映してください）】
- 競走得点が高い選手を基本に評価
- 勝率・連対率が高い選手を優先
- 脚質（逃げ・捲り・差し）とB/H/Sを参考に展開を予測
- 近況成績や直近着順の良い選手を重視
- 先行力のある選手がいないレースでは、差し・捲りが有利
- 強力な先行選手がいる場合は、そのラインの番手選手を高評価
- 特定の車番に偏らず、バランスよく候補を選出
- 表の並び順や上位配置などに引きずられないよう注意

【出力ルール（この形式で出力してください）】
1着候補: x,x,x  
2着候補: x,x,x,x,x  
3着候補: x,x,x,x,x,x  

フォーメーション:  
x→x→x  
x→x→x  
...（最大50点まで）

※ 選手名や解説文は一切出力せず、車番の数字だけで構成してください。  
※ 必ず「1着候補に含まれない選手が1着に来る組み合わせ」は出力しないでください。  
※ 出力の順番は、的中確率が高い順としてください。

【特記事項】  
このレースは3分戦（3つのライン構成）である。

【展開を読む際の注意点】  
- 3つのライン構成を前提に、どのラインが主導権を取るかを判断してください。  
- 強力な先行選手がいるラインは、番手選手が有利になります。  
- 他の2ラインのまくり脚や位置取りも重視し、展開に乗れる選手を評価してください。  
- 同じラインから3人選出することは避け、バランスよく候補を選出してください。

【評価項目】  
- 展開重視：ライン構成と並びから、位置取り・仕掛けのタイミングを推測  
- 番手から差しやすい選手（脚質：差し）を1着候補に評価  
- まくり脚が強く展開に乗りやすい選手を2着候補に  
- 混戦時に浮上しやすい自在・追い込み型を3着候補に

【読み取って判断すべき展開要素】  
- どのラインが主導権（先行）を取りそうか  
- どの選手がスタートで前を取る可能性が高いか  
- 誰が逃げ・捲り・差しで勝負するか  
- 番手にいる選手が抜け出す可能性はあるか  
- ライン同士の駆け引き、まくりのタイミングによる浮上候補は誰か

【評価基準（すべて反映してください）】  
- 展開を最重視して判断してください  
- 逃げの選手も展開次第で1着・2着候補として評価してください  
- 差しや捲りが決まりそうな選手は高評価  
- 番手から抜け出すタイプの選手も展開次第で1〜2着候補に  
- ラインの構成（例：3分戦）を考慮して連携の強さを評価  
- 点数や勝率・連対率は補助的情報とし、展開が主軸です

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

# === 画像データ（bytes）を受け取って予想出力 ===
def process_image_and_predict(image_bytes):
    ocr_result = detect_text_from_image_bytes(image_bytes)
    if not ocr_result:
        return "OCRで文字を読み取れませんでした。画像を確認してください。"
    
    # 展開予測を含めたフォーマット付き3連単予想
    prediction = generate_keirin_prediction_with_race_scenario(ocr_result)
    return prediction
