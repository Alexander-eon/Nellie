"""Тест: сохранение и загрузка чекпоинта."""
import sys
sys.path.insert(0, '.')
import numpy as np
from agent import Agent


def main():
    print("=== Тест 1: агент работает и сохраняется ===")
    agent = Agent(dt=0.05)
    agent.start()
    try:
        for i in range(100):
            agent.step()
    finally:
        agent.stop()

    print(f"Прожито тактов: {agent.step_count}")
    print(f"Финальная энергия: {agent.energy.energy:.4f}")
    print(f"Норма весов мозга: {np.linalg.norm(agent.brain.weights):.6f}")
    print()

    # Сохраняем
    path = agent.save("checkpoints/test.npz")
    print()

    # Запоминаем веса
    original_weights = agent.brain.weights.copy()

    print("=== Тест 2: создаём нового агента и загружаем ===")
    agent2 = Agent(dt=0.05)
    print(f"Новый агент, норма весов: {np.linalg.norm(agent2.brain.weights):.6f}")

    loaded = agent2.load("checkpoints/test.npz")
    print(f"Загрузка успешна: {loaded}")
    print(f"После загрузки, норма весов: {np.linalg.norm(agent2.brain.weights):.6f}")
    print()

    # Сравниваем
    diff = np.abs(agent2.brain.weights - original_weights).max()
    print(f"Максимальное расхождение весов: {diff:.10f}")

    if diff < 1e-10:
        print("✅ ЧЕКПОИНТ РАБОТАЕТ: веса совпадают точно")
    else:
        print("❌ ОШИБКА: веса не совпадают")


if __name__ == "__main__":
    main()
