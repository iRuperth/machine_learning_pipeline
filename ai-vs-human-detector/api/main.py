from fastapi import FastAPI, Request
import pickle
import sys
import os
from pathlib import Path
from fastapi.templating import Jinja2Templates
from model.features import extract_features
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, "model"))
import features
import __main__
__main__.pipeline_extractor = features.pipeline_extractor
__main__.extract_features = features.extract_features

app = FastAPI(title="AI vs Human Detector")
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR.parent / "templates"))
MODEL_PATH = os.path.join(project_root, "model", "detector_ia.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(text: str):
    if model is None:
        return {"error": "Model not found"}
        
    pred = model.predict([text])[0]
    prob = model.predict_proba([text])[0]

    return {
        "text": text,
        "prediction": "IA" if pred == 1 else "Human",
        "confidence": float(max(prob))
    }
