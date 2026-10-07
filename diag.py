"""Диагностика: ошибка, температура, веса на 1500 тактов."""
import sys
sys.path.insert(0, '.')
from agent import Agent
import numpy as np


def main():
    agent = Agent(dt=0.05, bounty_probability=1.0)
    agent.start()

    errors = []
    temps = []
    weights_norm = []

    try:
        for i in range(1500):
            log = agent.step()
            if log['step'] > 0:
                errors.append(log['error'])
                _, state = agent.sensor.read()
                temps.append(state.get('gpu_temp', 0))
                weights_norm.append(np.linalg.norm(agent.brain.weights))
    finally:
        agent.stop()

    errors = np.array(errors)
    temps = np.array(temps)
    weights_norm = np.array(weights_norm)

    print('=== ОШИБКА (по 100 тактов) ===')
    for i in range(0, len(errors), 100):
        print(f'  {i+100:>4}: err={errors[i:i+100].mean():.4f}')

    print()
    print('=== GPU TEMP ===')
    for i in range(0, len(temps), 100):
        print(f'  {i+100:>4}: temp={temps[i:i+100].mean():.3f}')

    print()
    print('=== |W| ===')
    for i in range(0, len(weights_norm), 100):
        print(f'  {i+100:>4}: |W|={weights_norm[i:i+100].mean():.4f}')


if __name__ == "__main__":
    main()
