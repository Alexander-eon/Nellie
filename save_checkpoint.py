import numpy as np
import pickle
from predictive_brain import PredictiveBrain

def save_brain(brain, filename):
    with open(filename, 'wb') as f:
        pickle.dump({'W_s': brain.W_s, 'W_a': brain.W_a}, f)
    print(f"Сохранено: {filename}")

def load_brain(brain, filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        brain.W_s = data['W_s']
        brain.W_a = data['W_a']
    return brain

if __name__ == "__main__":
    # Это просто заглушка для теста, в реальности мы бы сохранили веса из test_grammar.py
    print("Модуль сохранения готов. (Веса агентов сохраняются в test_grammar.py)")
