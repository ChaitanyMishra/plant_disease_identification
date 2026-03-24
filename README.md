# 🌿 Plant Disease Detection System
**BCA Major Project | Maharana Pratap College | Session 2025-26**

---

## 📁 Project Structure
```
plant_disease_project/
├── train.py              ← Run on Google Colab to train the AI model
├── app.py                ← Flask backend server
├── requirements.txt      ← Python dependencies
├── templates/
│   └── index.html        ← Frontend UI
└── model/                ← Put your trained model files here
    ├── plant_disease_model.h5   (generated after training)
    └── class_names.json         (generated after training)
```

---

## 🚀 Step-by-Step Setup Guide

### STEP 1 — Install Python dependencies (do this once)
Open Terminal on Linux Mint and run:
```bash
pip install tensorflow flask numpy Pillow --break-system-packages
```

---

### STEP 2 — Train the model on Google Colab (FREE GPU)

> ⚠️ Training requires a GPU. Use Google Colab for free.

1. Open **https://colab.research.google.com**
2. Click **New Notebook**
3. Go to **Runtime → Change Runtime Type → GPU** (select T4 GPU)
4. Click the **+ Code** button
5. Copy the entire contents of `train.py` and paste it
6. Click the **▶ Run** button
7. Wait ~30-45 minutes for training to complete
8. After training, download these two files from Colab's file browser (left sidebar):
   - `plant_disease_model.h5`
   - `class_names.json`
9. Place both files inside the `model/` folder in this project

---

### STEP 3 — Run the Web Application

Open Terminal in the project folder and run:
```bash
python app.py
```

You should see:
```
✅ Model loaded! Supports 38 disease classes.
🌿 Plant Disease Detection Server running!
   Open http://localhost:5000 in your browser
```

Open your browser and go to: **http://localhost:5000**

---

## 🎯 How to Use the App

1. Upload a photo of a plant leaf (JPG, PNG, or WEBP)
2. Click **"Analyse Leaf"**
3. The AI will detect the disease and show:
   - Disease name & plant species
   - Confidence percentage
   - Treatment recommendation
   - Top 3 possible diagnoses

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| AI Model  | MobileNetV2 (Transfer Learning) |
| Framework | TensorFlow / Keras |
| Dataset   | PlantVillage (38 disease classes) |
| Backend   | Python + Flask |
| Frontend  | HTML + CSS + JavaScript |

---

## 🌱 Supported Plant Diseases
The model detects **38 classes** across plants including:
- Tomato (Early Blight, Late Blight, Leaf Mold, etc.)
- Potato (Early Blight, Late Blight)
- Apple (Scab, Black Rot, Cedar Rust)
- Corn, Grape, Pepper, Strawberry, and more!

---

## 👥 Team
- Chaaitany Mishra (23071001635)
- Bhuyash Pathak (23071001633)
- Aryan Gupta (23071001601)
- Asmit Singh (23071001610)

**Guide:** Mr. Shiv Bahadur Singh
# plant_disease_identification
