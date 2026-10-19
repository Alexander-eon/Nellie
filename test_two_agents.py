"""
Два независимых Predictive-агента.
A управляет CPU, B управляет GPU. Пока без сообщений.
"""
import time
import numpy as np
from predictive_brain import PredictiveBrain
from body_sensor import BodySensor
from body_actuator import BodyActuator
from energy import Energy

def main():
    sensor = BodySensor()
    actuator = BodyActuator()
    energy = Energy(initial=1.0, drain_rate=0.002, bounty_rate=0.003,
                    cpu_min=0.45, cpu_max=0.55, gpu_min=0.45, gpu_max=0.55)

    # Агент A (CPU)
    brain_a = PredictiveBrain(n_sensors=1, n_actions=1, lr=0.05, lr_action=0.1)
    # Агент B (GPU)
    brain_b = PredictiveBrain(n_sensors=1, n_actions=1, lr=0.05, lr_action=0.1)

    actuator.start()
    time.sleep(1.0)

    target = np.array([0.5])
    
    print("Шаг | Жизнь | CPU | GPU | A_action | B_action")
    print("-" * 55)

    step = 0
    try:
        for step in range(1000):
            _, state = sensor.read()
            cpu = state.get("cpu_load", 0.0)
            gpu = state.get("gpu_load", 0.0)

            # Входы: только свои сенсоры
            input_a = np.array([cpu])
            input_b = np.array([gpu])

            # Действия
            action_a = brain_a.act(input_a, target)
            action_b = brain_b.act(input_b, target)

            # Шум для исследования (как в рабочем test_pred_agent)
            action_a = np.clip(action_a + np.random.normal(0.0, 0.02, size=1), 0.0, 1.0)
            action_b = np.clip(action_b + np.random.normal(0.0, 0.02, size=1), 0.0, 1.0)

            actuator.set_intensity(action_a[0], action_b[0])
            time.sleep(0.05)

            _, next_state = sensor.read()
            next_cpu = next_state.get("cpu_load", 0.0)
            next_gpu = next_state.get("gpu_load", 0.0)

            # Обучение
            brain_a.learn(input_a, action_a, np.array([next_cpu]))
            brain_b.learn(input_b, action_b, np.array([next_gpu]))

            energy.update(next_state)

            if step % 50 == 0:
                print(f"{step:3d} | {energy.energy:.3f} | {cpu:.2f} | {gpu:.2f} | {action_a[0]:.2f} | {action_b[0]:.2f}")

            if not energy.is_alive():
                print(f"*** Смерть на такте {step} ***")
                break

    finally:
        actuator.stop()
        print(f"Прожито тактов: {step}")

if __name__ == "__main__":
    main()
