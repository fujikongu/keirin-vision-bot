import os
from google.cloud import vision
from google.oauth2 import service_account

CREDENTIALS_PATH = "google_credentials.json"

def extract_text_from_image(image_path):
    credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
    client = vision.ImageAnnotatorClient(credentials=credentials)

    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if not texts:
        return "文字が検出されませんでした。"

    return texts[0].description.strip()
