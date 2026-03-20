import os
import cv2
import numpy as np
import pickle
import io
from flask import Flask, render_template, request, jsonify
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from PIL import Image

app = Flask(__name__)

# ==============================
# 1. Load trained objects
# ==============================
def load_models():
    try:
        with open("scaler_svm.pkl", "rb") as f:
            scl = pickle.load(f)
        with open("selector_svm.pkl", "rb") as f:
            sel = pickle.load(f)
        with open("fruit_model_svm (1).pkl", "rb") as f:
            mdl = pickle.load(f)
        return scl, sel, mdl
    except FileNotFoundError as e:
        print(f"Error: Model files not found! {e}")
        return None, None, None

scaler, selector, model = load_models()

# ==============================
# 2. Synchronized Parameters
# ==============================
TARGET_SIZE = (128, 128) 
LBP_POINTS = 24          
LBP_RADIUS = 3           
GLCM_DIST = 5            

CLASS_LABELS = {
    0: "Fresh Apple", 1: "Fresh Banana", 2: "Fresh Orange",
    3: "Rotten Apple", 4: "Rotten Banana", 5: "Rotten Orange"
}

# ==============================
# 3. Hybrid Feature Extraction
# ==============================
def extract_hybrid_features(img):
    # 1. Color features (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv],[0],None,[32],[0,180])
    hist_s = cv2.calcHist([hsv],[1],None,[32],[0,256])
    hist_v = cv2.calcHist([hsv],[2],None,[32],[0,256])
    
    color_hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    color_hist /= (color_hist.sum() + 1e-6)

    mean = np.mean(hsv, axis=(0,1))
    std  = np.std(hsv, axis=(0,1))
    color_stats = np.concatenate([mean, std])

    # 2. HOG (Shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(16,16),
                   cells_per_block=(2,2), block_norm='L2-Hys')

    # 3. LBP (Texture)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    # 4. GLCM
    glcm = graycomatrix(gray, distances=[GLCM_DIST], angles=[0], levels=256, symmetric=True, normed=True)
    glcm_features = np.array([
            graycoprops(glcm, 'contrast')[0,0],
            graycoprops(glcm, 'correlation')[0,0],
            graycoprops(glcm, 'energy')[0,0],
            graycoprops(glcm, 'homogeneity')[0,0]
        ])

    return np.concatenate([color_hist, color_stats, hog_feat, lbp_hist, glcm_features])

# ==============================
# 4. Flask Routes
# ==============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded on server'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read file
        img_bytes = file.read()
        
        # Try OpenCV decoding
        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        
        # If OpenCV fails (common with .avif), use PIL
        if img is None:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Pre-process & Predict
        img_resized = cv2.resize(img, TARGET_SIZE)
        feat = extract_hybrid_features(img_resized).reshape(1, -1)
        
        feat_scaled = scaler.transform(feat)
        feat_selected = selector.transform(feat_scaled)
        pred_index = model.predict(feat_selected)[0]
        
        result = CLASS_LABELS.get(int(pred_index), "Unknown")
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)