import os
import json
from google.cloud import vision
from google.oauth2 import service_account
from PIL import Image
import io

def load_credentials_from_env():
    """
    環境変数 GOOGLE_CREDENTIALS_JSON からサービスアカウント認証情報を読み込む
    """
    try:
        credentials_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
        if not credentials_json:
            raise Exception("環境変数 GOOGLE_CREDENTIALS_JSON が設定されていません。")
        credentials_info = json.loads(credentials_json)
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        return credentials
    except Exception as e:
        print(f"認証情報の読み込みに失敗しました: {e}")
        raise

def detect_text_from_image(image_path: str) -> str:
    """
    指定された画像ファイルからテキストを抽出して返す
    """
    credentials = load_credentials_from_env()
    client = vision.ImageAnnotatorClient(credentials=credentials)

    try:
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations

        if response.error.message:
            raise Exception(f'OCR APIエラー: {response.error.message}')

        if texts:
            return texts[0].description.strip()
        else:
            return "テキストが検出されませんでした。"

    except Exception as e:
        return f"OCR解析中にエラーが発生しました: {e}"

# テスト実行例（開発中のみ使用）
if __name__ == "__main__":
    test_image_path = "test_keirin_image.png"  # 任意のローカル画像パス
    result = detect_text_from_image(test_image_path)
    print("OCR結果:\n", result)
