from fastapi import FastAPI
import pickle
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, "model"))

from features import extract_features

app = FastAPI(title="Detector IA vs Humano")

MODEL_PATH = os.path.join(project_root, "model", "detector_ia.pkl")

# Carga del modelo
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"ERROR: No se encontró el modelo en {MODEL_PATH}")
    model = None

@app.get("/")
def home():
    return {"status": "Working", "model_loaded": model is not None}

@app.post("/predict")
def predict(text: str):
    if model is None:
        return {"error": "Modelo no cargado correctamente en el servidor"}
        
    features = extract_features(text)
    features_array = np.array([features])

    pred = model.predict(features_array)[0]
    prob = model.predict_proba(features_array)[0]

    return {
        "text": text,
        "prediction": "IA" if pred == 1 else "Humano",
        "confidence": float(max(prob))
    }