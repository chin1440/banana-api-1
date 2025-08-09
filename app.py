import os
from io import BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from PIL import Image
from torchvision import transforms
import torch

APP = Flask(__name__)
CORS(APP)

# ---------- CONFIG ----------
MODEL_PATH = "banana_classifier.pt"
# ใส่ไฟล์ .pt ไว้บน Google Drive แล้วตั้งค่า Environment variable ชื่อ MODEL_FILE_ID บน Render
FILE_ID = os.environ.get("MODEL_FILE_ID")

CLASSES = [
    "Banana Black Sigatoka",
    "Banana Bract Mosaic Virus",
    "Banana Insect Pest",
    "Banana Moko",
    "Banana Panama",
    "Banana Yellow Sigatoka",
    "Healthy",
]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
# ----------------------------

def _ensure_model():
    """ถ้าไม่มีไฟล์โมเดลในดิสก์ ให้โหลดจาก Google Drive (ต้องตั้ง MODEL_FILE_ID)"""
    if os.path.exists(MODEL_PATH):
        return
    if not FILE_ID:
        raise RuntimeError("MODEL_FILE_ID is not set and model file is missing.")
    import gdown
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

_ensure_model()
MODEL = torch.load(MODEL_PATH, map_location="cpu")
MODEL.eval()

def _infer_from_url(image_url: str) -> str:
    resp = requests.get(image_url, timeout=20)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        y = MODEL(x)
        idx = int(y.argmax(1).item())
    return CLASSES[idx] if 0 <= idx < len(CLASSES) else str(idx)

@APP.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@APP.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        image_url = request.args.get("image_url")
    else:
        data = request.get_json(silent=True) or {}
        image_url = data.get("image_url")

    if not image_url:
        return jsonify({"error": "Missing image_url"}), 400

    try:
        result = _infer_from_url(image_url)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render จะส่ง PORT มาทาง env
    APP.run(host="0.0.0.0", port=port)
