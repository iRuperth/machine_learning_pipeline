import re
import numpy as np

# Analizamos el texto para sacar números que nos den pistas de quien escribió el mismo.
def extract_features(text):
    words = text.split()
    total_length = len(text)
    num_words = len(words)
    unique_words = len(set(words))

    # Promedio de largo de palabra.
    avg_word_length = np.mean([len(w) for w in words]) if words else 0
    
    # Contamos signos de puntuación.
    punctuation = len(re.findall(r'[.,;:!?]', text))

    # Cantidad de vocales y letras.
    vowels = len(re.findall(r'[aeiouáéíóúAEIOU]', text))
    vowel_ratio = vowels / total_length if total_length > 0 else 0
    alphas = sum(c.isalpha() for c in text)
    alpha_ratio = alphas / total_length if total_length > 0 else 0
    
    # Variedad del vocabulario.
    lexical_diversity = unique_words / num_words if num_words > 0 else 0

    # Control de mayúsculas.
    caps_count = sum(1 for c in text if c.isupper())
    caps_ratio = caps_count / total_length if total_length > 0 else 0

    # Pasamos los datos limpios en una lista para que el modelo los entienda
    return [
        total_length,
        num_words,
        unique_words,
        avg_word_length,
        punctuation,
        vowel_ratio,
        alpha_ratio,
        lexical_diversity,
        caps_ratio
    ]

# Importante! // aqui se procesa una lista entera de textos de una sola vez
def pipeline_extractor(texts):
    return np.array([extract_features(t) for t in texts])

# Tips con Pickle:
# Forzamos que estas funciones se identifiquen siempre como parte de "features"
# Así la API las encuentra sin problemas aunque el modelo se haya entrenado en otro lado
pipeline_extractor.__module__ = "features"
extract_features.__module__ = "features"
