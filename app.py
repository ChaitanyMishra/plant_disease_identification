from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io
import json
import os

import requests

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
CONFIDENCE_THRESHOLD = 70.0

try:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
except:
    mobilenet_preprocess = None

app = Flask(__name__)

model       = None
class_names = []
LAST_PREDICTION_CONTEXT = {}
PROJECT_CONTEXT = {
    "project_name": "LeafAI",
    "built_by": ["Chaitany Mishra", "Aryan Gupta", "Bhuyash Pathak", "Asmit Singh"],
    "college": "Maharana Pratap College",
    "session": "2025-26",
    "stack": ["Flask", "TensorFlow", "MobileNetV2", "PlantVillage Dataset", "Gemini API"],
    "how_it_works": [
        "User uploads a leaf image",
        "Model predicts plant disease and confidence",
        "If confidence is below 70%, Gemini fallback analyzes image",
        "System returns diagnosis, treatment, and 7-day routine",
        "Chatbot answers both plant and project/website questions using Gemini"
    ]
}

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


# Treatment advice (short)
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

# 7-day routines for each disease
ROUTINES = {
    'healthy': [
        "Day 1: Continue regular watering and monitor leaves.",
        "Day 2: Check for pests or unusual spots.",
        "Day 3: Ensure proper sunlight and airflow.",
        "Day 4: Fertilize if needed.",
        "Day 5: Remove any debris around the plant.",
        "Day 6: Inspect for early disease symptoms.",
        "Day 7: Enjoy your healthy plant!"
    ],
    'early blight': [
        "Day 1: Remove infected lower leaves and dispose of them.",
        "Day 2: Apply copper-based fungicide to affected areas.",
        "Day 3: Avoid overhead watering; water at the base.",
        "Day 4: Monitor for new spots and remove if found.",
        "Day 5: Improve air circulation around the plant.",
        "Day 6: Reapply fungicide if needed.",
        "Day 7: Inspect and repeat steps if symptoms persist."
    ],
    'late blight': [
        "Day 1: Remove and destroy infected leaves and stems.",
        "Day 2: Apply fungicide thoroughly.",
        "Day 3: Avoid wetting foliage; water early in the day.",
        "Day 4: Check for spread and remove new infections.",
        "Day 5: Reapply fungicide if rain occurs.",
        "Day 6: Ensure good drainage and airflow.",
        "Day 7: Continue monitoring and repeat as needed."
    ],
    'leaf mold': [
        "Day 1: Remove affected leaves and dispose of them.",
        "Day 2: Reduce humidity and increase ventilation.",
        "Day 3: Apply mancozeb fungicide.",
        "Day 4: Avoid overhead watering.",
        "Day 5: Monitor for new mold spots.",
        "Day 6: Reapply fungicide if necessary.",
        "Day 7: Inspect and maintain dry conditions."
    ],
    'bacterial spot': [
        "Day 1: Remove and destroy infected leaves.",
        "Day 2: Apply copper-based bactericide.",
        "Day 3: Avoid overhead irrigation.",
        "Day 4: Monitor for new spots and remove if found.",
        "Day 5: Disinfect tools after use.",
        "Day 6: Reapply bactericide if needed.",
        "Day 7: Continue monitoring and repeat as needed."
    ],
    'mosaic virus': [
        "Day 1: Remove infected plants immediately.",
        "Day 2: Control aphids and other vectors.",
        "Day 3: Disinfect tools and hands after handling.",
        "Day 4: Monitor nearby plants for symptoms.",
        "Day 5: Remove weeds that may host the virus.",
        "Day 6: Apply insecticidal soap if aphids persist.",
        "Day 7: Continue monitoring and remove new infections."
    ],
    'yellow leaf curl': [
        "Day 1: Remove heavily infected leaves.",
        "Day 2: Control whitefly population with insecticide.",
        "Day 3: Use yellow sticky traps for monitoring.",
        "Day 4: Inspect for new symptoms and remove affected parts.",
        "Day 5: Apply insecticide again if needed.",
        "Day 6: Ensure good plant nutrition.",
        "Day 7: Continue monitoring and repeat as needed."
    ],
    'default': [
        "Day 1: Remove infected parts and dispose of them.",
        "Day 2: Apply appropriate fungicide or bactericide.",
        "Day 3: Monitor for new symptoms.",
        "Day 4: Improve air circulation and reduce humidity.",
        "Day 5: Reapply treatment if needed.",
        "Day 6: Consult a local agricultural expert.",
        "Day 7: Continue monitoring and repeat as needed."
    ]
}
def get_routine(disease_name):
    d = disease_name.lower()
    for key, routine in ROUTINES.items():
        if key in d:
            return routine
    return ROUTINES['default']

def get_treatment(disease_name):
    d = disease_name.lower()
    for key, advice in TREATMENTS.items():
        if key in d:
            return advice
    return TREATMENTS['default']


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded yet.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

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

    gemini_result = None
    gemini_used = False
    if confidence < CONFIDENCE_THRESHOLD:
        try:
            from gemini_api import gemini_analyze_leaf_health
            gemini_result = gemini_analyze_leaf_health(img_bytes, GEMINI_API_KEY)
            gemini_used = True
            # Low-confidence fallback: switch output to Gemini result when available.
            if isinstance(gemini_result, dict):
                gemini_plant = (gemini_result.get('plant_name') or '').strip()
                gemini_disease = (gemini_result.get('disease_name') or '').strip()
                if gemini_plant:
                    plant_name = gemini_plant
                if gemini_disease:
                    disease_name = gemini_disease
                gemini_healthy = gemini_result.get('is_healthy')
                if isinstance(gemini_healthy, bool):
                    is_healthy = gemini_healthy
                gemini_advice = gemini_result.get('detailed_advice', '').strip()
                if gemini_advice:
                    treatment = gemini_advice
        except Exception as e:
            gemini_result = f"Gemini API error: {e}"
            gemini_used = False
    routine = get_routine(disease_name)
    global LAST_PREDICTION_CONTEXT
    LAST_PREDICTION_CONTEXT = {
        'plant': plant_name,
        'disease': disease_name,
        'confidence': round(confidence, 1),
        'is_healthy': is_healthy,
        'treatment': treatment,
        'top3': top3,
        'gemini_used': gemini_used,
        'gemini_result': gemini_result
    }
    return jsonify({
        'success':    True,
        'plant':      plant_name,
        'disease':    disease_name,
        'confidence': round(confidence, 1),
        'is_healthy': is_healthy,
        'treatment':  treatment,
        'top3':       top3,
        'gemini_result': gemini_result,
        'gemini_used': gemini_used,
        'routine':    routine
    })


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json() or {}
        user_message = (data.get('message') or '').strip()
        if not user_message:
            return jsonify({'error': 'Message is required.'}), 400
        prediction_context = data.get('prediction') or LAST_PREDICTION_CONTEXT

        from gemini_api import gemini_chat_response
        reply = gemini_chat_response(user_message, prediction_context, PROJECT_CONTEXT, GEMINI_API_KEY)
        return jsonify({'reply': reply})
    except Exception as e:
        return jsonify({'error': f'Chat failed: {str(e)}'}), 500

# Endpoint to get 7-day routine for a disease
@app.route('/routine', methods=['POST'])
def routine():
    try:
        data = request.get_json()
        disease = data.get('disease', '')
        routine = get_routine(disease)
        return jsonify({'routine': routine})
    except Exception as e:
        return jsonify({'error': f'Routine fetch failed: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None, 'num_classes': len(class_names)})

if __name__ == '__main__':
    load_model()
    print("\n🌿 Server running! Open http://localhost:5000\n")
    app.run(debug=False, port=5000, host='0.0.0.0', use_reloader=False)