import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import numpy as np

# Traemos la lógica de extracción de nuestro archivo features / características.
from features import pipeline_extractor

# Configuramos las rutas de los datos y dónde guardaremos el modelo final
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "../data/dataset.csv"
MODEL_OUTPUT = BASE_DIR / "detector_ia.pkl"

# Cargamos el dataset con los textos y sus etiquetas (IA o Humano)
df = pd.read_csv(DATA_PATH)

# Dividimos los datos para entrenar con una parte y probar con la otra
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# Creamos el Pipeline oficial: primero extrae las pistas y luego clasifica
# Usamos FunctionTransformer para que nuestra función sea parte del flujo
pipeline = Pipeline([
    ('extractor', FunctionTransformer(pipeline_extractor)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Entrenamos el pipeline completo (procesamiento + modelo) de una sola vez
pipeline.fit(X_train, y_train)

# Verificamos qué tan bien aprendió el modelo con los datos de prueba
pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, pred):.2%}")

# Guardamos todo el "tubo" de trabajo en un archivo .pkl usando pickle
# Así la API podrá usarlo sin tener que configurar nada manualmente
with open(MODEL_OUTPUT, "wb") as f:
    pickle.dump(pipeline, f)

print("Saved pipeline successfully")
