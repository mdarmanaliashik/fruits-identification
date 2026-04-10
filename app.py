import os
import cv2
import pickle
import numpy as np
from flask import Flask, request, render_template
from rembg import remove
from skimage.feature import hog, local_binary_pattern

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fruit_model_xgboost.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler_xgboost.pkl")

# =========================
# Load model and scaler
# =========================
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# =========================
# Class names
# =========================
class_names = [
    "Fresh Apple",
    "Fresh Banana",
    "Fresh Orange",
    "Rotten Apple",
    "Rotten Banana",
    "Rotten Orange"
]

# =========================
# Fruit details
# =========================
fruit_details = {
    "Fresh Apple": {
        "info": "Apple is a nutritious fruit rich in fiber, vitamins, and antioxidants.",
        "effect": "Eating fresh apple may support digestion, improve overall health, and provide natural energy.",
        "advice": "Wash properly before eating."
    },
    "Fresh Banana": {
        "info": "Banana is an energy-rich fruit containing potassium, vitamin B6, and carbohydrates.",
        "effect": "Eating fresh banana may provide quick energy, support muscle function, and help digestion.",
        "advice": "Eat when the peel is yellow and the fruit is firm."
    },
    "Fresh Orange": {
        "info": "Orange is a citrus fruit rich in vitamin C, antioxidants, and water.",
        "effect": "Eating fresh orange may support immunity, hydration, and skin health.",
        "advice": "Peel and wash properly before eating."
    },
    "Rotten Apple": {
        "info": "This apple appears spoiled and may contain decay or harmful microbial growth.",
        "effect": "Eating rotten apple may cause stomach discomfort, nausea, vomiting, or diarrhea.",
        "advice": "Do not eat spoiled apple. Discard it immediately."
    },
    "Rotten Banana": {
        "info": "This banana appears spoiled, over-fermented, or contaminated.",
        "effect": "Eating rotten banana may cause nausea, vomiting, abdominal pain, or diarrhea.",
        "advice": "Avoid consuming rotten banana and throw it away."
    },
    "Rotten Orange": {
        "info": "This orange appears spoiled and may contain mold or internal decay.",
        "effect": "Eating rotten orange may lead to stomach upset, vomiting, diarrhea, or food poisoning symptoms.",
        "advice": "Do not consume rotten or moldy orange."
    }
}

# =========================
# Config
# =========================
IMG_SIZE = (128, 128)
LBP_POINTS = 24
LBP_RADIUS = 3

sift = cv2.SIFT_create(
    nfeatures=1000,
    nOctaveLayers=4,
    contrastThreshold=0.02,
    edgeThreshold=15,
    sigma=1.6
)

# =========================
# Background removal
# =========================
def remove_background_keep_black(img):
    output = remove(img)

    if len(output.shape) == 3 and output.shape[2] == 4:
        item_mask = output[:, :, 3]
        black_bg = np.zeros_like(img)
        final_img = np.where(item_mask[:, :, None] > 0, img, black_bg)
        return final_img

    return img

# =========================
# Preprocessing
# =========================
def preprocess_image_from_bytes(file_bytes):
    file_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    if img is None:
        return None

    img = remove_background_keep_black(img)
    img = cv2.resize(img, IMG_SIZE)

    return img

# =========================
# Feature extraction
# =========================
def extract_features_from_image(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])

    color_hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    color_hist = color_hist / (color_hist.sum() + 1e-6)

    mean = np.mean(hsv, axis=(0, 1))
    std = np.std(hsv, axis=(0, 1))
    color_stats = np.concatenate([mean, std])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

    lbp = local_binary_pattern(
        gray,
        LBP_POINTS,
        LBP_RADIUS,
        method="uniform"
    )

    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, LBP_POINTS + 3),
        range=(0, LBP_POINTS + 2)
    )

    lbp_feat = lbp_hist.astype(np.float32)
    lbp_feat = lbp_feat / (np.sum(lbp_feat) + 1e-7)

    _, des = sift.detectAndCompute(gray, None)

    if des is not None:
        sift_feat = np.mean(des, axis=0).astype(np.float32)
    else:
        sift_feat = np.zeros(128, dtype=np.float32)

    hybrid_vector = np.concatenate([
        color_hist,
        color_stats,
        hog_feat,
        lbp_feat,
        sift_feat
    ]).astype(np.float32)

    return hybrid_vector

# =========================
# Prediction
# =========================
def predict_image(img):
    feat = extract_features_from_image(img)
    feat = feat.reshape(1, -1)

    feat_scaled = scaler.transform(feat)
    pred = model.predict(feat_scaled)[0]

    if isinstance(pred, (int, np.integer)):
        idx = int(pred)
        if 0 <= idx < len(class_names):
            return class_names[idx]

    pred_str = str(pred)
    if pred_str in class_names:
        return pred_str

    try:
        idx = int(float(pred_str))
        if 0 <= idx < len(class_names):
            return class_names[idx]
    except Exception:
        pass

    return pred_str

# =========================
# Routes
# =========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return render_template("result.html", error="No image uploaded.")

        file = request.files["image"]

        if file.filename == "":
            return render_template("result.html", error="No image selected.")

        file_bytes = file.read()
        img = preprocess_image_from_bytes(file_bytes)

        if img is None:
            return render_template("result.html", error="Invalid image file.")

        prediction = predict_image(img)

        details = fruit_details.get(prediction, {
            "info": "No information available.",
            "effect": "No effect information available.",
            "advice": "No advice available."
        })

        is_fresh = "fresh" in prediction.lower()

        return render_template(
            "result.html",
            prediction=prediction,
            info=details["info"],
            effect=details["effect"],
            advice=details["advice"],
            is_fresh=is_fresh
        )

    except Exception as e:
        print("ERROR:", str(e))
        return render_template("result.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
