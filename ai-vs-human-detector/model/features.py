import re
import numpy as np

def extract_features(text):
    words = text.split()
    total_length = len(text)
    num_words = len(words)
    unique_words = len(set(words))

    avg_word_length = np.mean([len(w) for w in words]) if words else 0
    punctuation = len(re.findall(r'[.,;:!?]', text))

    vowels = len(re.findall(r'[aeiouáéíóúAEIOU]', text))
    vowel_ratio = vowels / total_length if total_length > 0 else 0
    alphas = sum(c.isalpha() for c in text)
    alpha_ratio = alphas / total_length if total_length > 0 else 0
    lexical_diversity = unique_words / num_words if num_words > 0 else 0

    caps_count = sum(1 for c in text if c.isupper())
    caps_ratio = caps_count / total_length if total_length > 0 else 0

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