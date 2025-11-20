import os, io, json, base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(ROOT, "models", "food_model.h5")
LABELS_PATH = os.path.join(ROOT, "models", "labels.json")
NUTRI_PATH = os.path.join(ROOT, "nutrients", "label_to_nutrients.json")
IMG_SIZE = (224, 224)

app = Flask(__name__)
CORS(app)

model = load_model(MODEL_PATH)
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)    # {"0":"pizza", ...}
idx_to_name = {int(k): v for k, v in labels.items()}
with open(NUTRI_PATH, "r", encoding="utf-8") as f:
    nutrit = json.load(f)

def preprocess_pil(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, 0)

def compute_nutrition(label, serving_g):
    info = nutrit.get(label)
    if not info:
        return None
    factor = float(serving_g) / 100.0
    per_serving = {
        "calories_kcal": round(info.get("calories_kcal_per_100g", 0) * factor, 1),
        "protein_g": round(info.get("protein_g_per_100g", 0) * factor, 1),
        "carbs_g": round(info.get("carbohydrates_g_per_100g", 0) * factor, 1),
        "fat_g": round(info.get("fat_g_per_100g", 0) * factor, 1),
        "fiber_g": round(info.get("fiber_g_per_100g", 0) * factor, 1),
    }
    return {"serving_g": serving_g, "per_serving": per_serving}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        serving_g = 100
        img = None

        if request.is_json:
            data = request.get_json()
            serving_g = float(data.get("serving_g", serving_g))
            b64 = data.get("image_base64") or ""
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            if b64:
                img = Image.open(io.BytesIO(base64.b64decode(b64)))
        else:
            if "file" in request.files:
                f = request.files["file"]
                img = Image.open(f.stream)
            elif "image_base64" in request.form:
                b64 = request.form["image_base64"]
                if "," in b64: b64 = b64.split(",",1)[1]
                img = Image.open(io.BytesIO(base64.b64decode(b64)))
            if "serving_g" in request.form:
                serving_g = float(request.form["serving_g"])

        if img is None:
            return jsonify({"error": "No image provided"}), 400

        arr = preprocess_pil(img)
        preds = model.predict(arr)[0]
        top_idxs = preds.argsort()[::-1][:3]
        out = []
        for i in top_idxs:
            name = idx_to_name.get(int(i), str(i))
            conf = float(preds[int(i)])
            nut = compute_nutrition(name, serving_g)
            out.append({"label": name, "confidence": round(conf,4), "nutrition": nut})

        return jsonify({"predictions": out})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
