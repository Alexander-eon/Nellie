"""
Тест PredictiveBrain на простой линейной среде.
Проверяем: 1) падает ли ошибка предсказания, 2) адекватны ли действия.
"""

import numpy as np
from predictive_brain import PredictiveBrain

def main():
    np.random.seed(42)
    n_sensors = 4
    n_actions = 2
    
    brain = PredictiveBrain(n_sensors, n_actions, lr=0.01, lr_action=0.5)
    
    # Простая среда: следующее состояние = 0.9 * текущее + A * действие
    # Матрица A (4x2): как 2 действия влияют на 4 сенсора
    A = np.array([
        [0.5, 0.0],
        [0.5, 0.0],
        [0.0, 0.5],
        [0.0, 0.5]
    ])
    def env_step(s, a):
        # В реальности загрузка CPU/GPU всегда от 0 до 1
        return np.clip(0.9 * s + A @ a, 0.0, 1.0)
    
    # 1. Тест обучения модели мира
    print("=== Тест 1: обучение предсказанию ===")
    errors = []
    for i in range(1000):
        # Каждый раз берем СЛУЧАЙНОЕ состояние и действие (а не одну траекторию)
        s = np.random.rand(n_sensors) 
        a = np.random.rand(n_actions)
        next_s = env_step(s, a)
        err = brain.learn(s, a, next_s)
        errors.append(err)
        
        if i % 200 == 0:
            print(f"  Шаг {i}: ошибка = {err:.6f}")
            
    print(f"  Итог: ошибка упала с {errors[0]:.6f} до {np.mean(errors[-50:]):.6f}")
    print()
    
    # 2. Тест выбора действия (Active Inference)
    print("=== Тест 2: выбор действия для достижения цели ===")
    # Сбрасываем среду в ноль
    s = np.zeros(n_sensors)
    target = np.array([0.5, 0.5, 0.5, 0.5])  # Достижимая цель (a=[1,1])
    
    action = brain.act(s, target)
    pred = brain.predict(s, action)
    print(f"  Цель:    {target}")
    print(f"  Действие: {action}")
    print(f"  Предсказание: {pred}")
    print(f"  Ошибка до цели: {np.mean(np.abs(pred - target)):.6f}")
    
    if np.mean(np.abs(pred - target)) < 0.1:
        print("  ✅ Действие адекватно: мозг выбрал действие, ведущее к цели")
    else:
        print("  ⚠️ Действие не идеально (модель ещё не обучена на всех состояниях)")

if __name__ == "__main__":
    main()
