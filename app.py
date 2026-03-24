from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io
import json
import os

try:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
except:
    mobilenet_preprocess = None

app = Flask(__name__)

model       = None
class_names = []

MODEL_PATH       = os.path.join('model', 'plant_disease_model.keras')
CLASS_NAMES_PATH = os.path.join('model', 'class_names.json')

def load_model():
    global model, class_names
    if not os.path.exists(MODEL_PATH):
        print("⚠️  Model file not found at:", MODEL_PATH)
        return
    try:
        import tensorflow as tf
        print("Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
        print(f"✅ Model loaded! Supports {len(class_names)} disease classes.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224), Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    if mobilenet_preprocess:
        img_array = mobilenet_preprocess(img_array)
    else:
        img_array = img_array / 127.5 - 1.0
    return np.expand_dims(img_array, axis=0)

def format_class_name(raw_name):
    parts = raw_name.replace('___', '|||').replace('_', ' ').split('|||')
    if len(parts) == 2:
        return parts[0].strip().title(), parts[1].strip().title()
    return raw_name.replace('_', ' ').title(), 'Unknown'

TREATMENTS = {
    'healthy':         "Your plant looks healthy! Keep watering regularly and monitor for early signs of disease.",
    'early blight':    "Apply copper-based fungicide. Remove infected lower leaves. Avoid overhead watering.",
    'late blight':     "Apply fungicide immediately. Remove and destroy infected parts. Avoid wet foliage.",
    'leaf mold':       "Improve air circulation. Reduce humidity. Apply fungicide (mancozeb).",
    'bacterial spot':  "Apply copper-based bactericide. Avoid overhead irrigation.",
    'mosaic virus':    "No cure available. Remove infected plants. Control aphids (common vector).",
    'yellow leaf curl':"Control whitefly population. Use insecticide if needed.",
    'default':         "Consult a local agricultural expert. Remove infected parts and apply appropriate fungicide."
}

def get_treatment(disease_name):
    d = disease_name.lower()
    for key, advice in TREATMENTS.items():
        if key in d:
            return advice
    return TREATMENTS['default']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded yet.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    try:
        img_bytes     = file.read()
        img_array     = preprocess_image(img_bytes)
        predictions   = model.predict(img_array, verbose=0)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence    = float(predictions[0][predicted_idx]) * 100
        raw_class     = class_names[predicted_idx]
        plant_name, disease_name = format_class_name(raw_class)
        is_healthy    = 'healthy' in disease_name.lower()
        treatment     = get_treatment(disease_name)

        top3_indices = np.argsort(predictions[0])[-3:][::-1]
        top3 = []
        for idx in top3_indices:
            p, d = format_class_name(class_names[idx])
            top3.append({'plant': p, 'disease': d, 'confidence': round(float(predictions[0][idx]) * 100, 1)})

        return jsonify({
            'success':    True,
            'plant':      plant_name,
            'disease':    disease_name,
            'confidence': round(confidence, 1),
            'is_healthy': is_healthy,
            'treatment':  treatment,
            'top3':       top3
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None, 'num_classes': len(class_names)})

if __name__ == '__main__':
    load_model()
    print("\n🌿 Server running! Open http://localhost:5000\n")
    app.run(debug=False, port=5000, host='0.0.0.0', use_reloader=False)