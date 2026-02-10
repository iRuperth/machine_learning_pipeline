from fastapi import FastAPI
import pickle
import numpy as np
import sys
import os
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pathlib import Path


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, "model"))

from features import extract_features


app = FastAPI(title="AI vs Human Detector")
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR.parent / "templates"))
MODEL_PATH = os.path.join(project_root, "model", "detector_ia.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"ERROR: Model not found {MODEL_PATH}")
    model = None

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict(text: str):
    if model is None:
        return {"error": "Error model not found"}
        
    features = extract_features(text)
    features_array = np.array([features])

    pred = model.predict(features_array)[0]
    prob = model.predict_proba(features_array)[0]

    return {
        "text": text,
        "prediction": "IA" if pred == 1 else "Human",
        "confidence": float(max(prob))
    }



# AI, human or uncertain

# @app.post("/predict")
# def predict(text: str):
#     if model is None:
#         return {"error": "Error model not found"}
        
#     features = extract_features(text)
#     features_array = np.array([features])

#     prob = model.predict_proba(features_array)[0]
#     prob_human = float(prob[0])
#     prob_ai = float(prob[1])

#     threshold = 0.70

#     if prob_ai >= threshold:
#         prediction = "IA"
#         confidence = prob_ai
#     elif prob_human >= threshold:
#         prediction = "Human"
#         confidence = prob_human
#     else:
#         prediction = "Uncertain"
#         confidence = max(prob_ai, prob_human)

#     return {
#         "text": text,
#         "prediction": prediction,
#         "confidence": confidence,
#         "ai_score": prob_ai,
#         "human_score": prob_human
#     }