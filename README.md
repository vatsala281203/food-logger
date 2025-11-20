# AI-Based Food Logging via Image & Nutrient Estimation

This project implements an end-to-end AI system that identifies food items from webcam images and estimates their nutritional values (calories, protein, carbohydrates, fat, and fiber).  
All inference runs **locally**.

---

## 1. Overview

### ✔ Food Classification  
A custom image classifier was trained locally using **MobileNetV2** (transfer learning).  
The model predicts one food label among **90 - Indian + global food classes**.

### ✔ Nutrition Estimation  
Each class has a per-100g nutrition mapping.  
For any serving size (default 100g), the system computes:

- Calories (kcal)  
- Protein (g)  
- Carbohydrates (g)  
- Fat (g)  
- Fiber (g)

### ✔ Web-Based Interface  
A simple frontend allows users to:

- Open webcam  
- Capture an image  
- Run prediction  
- View top label and confidence  
- View nutritional breakdown  

### ✔ Local Backend (Flask)
The backend performs:

- Image preprocessing  
- TensorFlow inference  
- Nutrition lookup  
- Returns JSON predictions  

---

## 2. Dataset Sources

This project uses a mixture of two open-source datasets:

- Indian food dataset: [Food Image Classification (Kaggle)](https://www.kaggle.com/datasets/gauravduttakiit/food-image-classification/data)  
- Global food dataset: [Food41 / Food-101 (Kaggle)](https://www.kaggle.com/datasets/kmader/food41/data?select=images)

Both datasets were reorganized into:

- **data/train/<class>/**
- **data/val/<class>/**
- **data/test/<class>/**


## 3. Model Training Summary

**Base model:** MobileNetV2 (feature extractor, ImageNet weights)  
**Head:** GAP → Dropout → Dense(512) → Dense(90) softmax  
**Input size:** 224×224  
**Epochs:** 10  
**Optimizer:** Adam (1e-4)  

### ✔ Evaluation Performance  
- **Weighted accuracy:** ~76%  
- **Macro-average accuracy:** ~55% 

---

## 4. System Workflow

### **Step 1 — User Captures Image (Webcam)**  
The image is sent from the browser to the Flask backend.

### **Step 2 — Backend Predicts Food Class**  
Model outputs prediction with confidence.

### **Step 3 — Nutrition Retrieval**  
Nutrition info is loaded from:

nutrients/label_to_nutrients.json

### **Step 4 — Display Results**  
Frontend displays:

- Predicted food  
- Confidence  
- Nutrition per serving  

---

## 5. How to Run the Project

### 1. Install Python dependencies
pip install -r requirements.txt

### 2. Start backend server
python src/backend/app.py

Backend runs on:
➡ http://127.0.0.1:5000

### 3. Start frontend
cd src/frontend
python -m http.server 8000

Frontend opens at:
➡ http://localhost:8000  

---