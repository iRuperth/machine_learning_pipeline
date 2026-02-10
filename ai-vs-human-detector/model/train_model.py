import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from features import extract_features

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "../data/dataset.csv"
MODEL_OUTPUT = BASE_DIR / "detector_ia.pkl"

# Cargar dataset
df = pd.read_csv(DATA_PATH)

# Crear features
X = df["text"].apply(extract_features).tolist()
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modelo
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluación
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print(f"Accuracy: {acc:.2%}")

# Guardado del modelo
with open(MODEL_OUTPUT, "wb") as f:
    pickle.dump(model, f)

print("Modelo guardado")
