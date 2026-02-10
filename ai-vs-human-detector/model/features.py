import re
import numpy as np

def extract_features(text):
    words = text.split()

    total_length = len(text)
    num_words = len(words)
    unique_words = len(set(words))

    avg_word_length = np.mean([len(w) for w in words]) if words else 0

    punctuation = len(re.findall(r'[.,;:!?]', text))

    return [
        total_length,
        num_words,
        unique_words,
        avg_word_length,
        punctuation
    ]
