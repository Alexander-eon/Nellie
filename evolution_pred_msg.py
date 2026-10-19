"""
Predictive-агенты с каналом сообщений. Асимметрия информации.
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

    # A: 2 входа (cpu, msg_b), 2 выхода (action_cpu, msg_a)
    brain_a = PredictiveBrain(n_sensors=2, n_actions=2, lr=0.05, lr_action=0.5)
    # B: 2 входа (gpu, msg_a), 2 выхода (action_gpu, msg_b)
    brain_b = PredictiveBrain(n_sensors=2, n_actions=2, lr=0.05, lr_action=0.5)

    actuator.start()
    time.sleep(1.0)

    target_cpu = 0.5
    target_gpu = 0.5
    
    last_msg_a = 0.0
    last_msg_b = 0.0

    print("Шаг | Жизнь | CPU | GPU | msg_a | msg_b")
    print("-" * 45)

    step = 0
    try:
        for step in range(1000):
            _, state = sensor.read()
            cpu = state.get("cpu_load", 0.0)
            gpu = state.get("gpu_load", 0.0)

            # Входы: только свой сенсор и сообщение от партнера
            input_a = np.array([cpu, last_msg_b])
            input_b = np.array([gpu, last_msg_a])

            # Действия: цель для A - свой CPU (target_cpu), цель для B - свой GPU (target_gpu)
            # Второй элемент цели - сообщение (0.5, чтобы не залипало на 0 или 1)
            action_a = brain_a.act(input_a, np.array([target_cpu, 0.5]))
            action_b = brain_b.act(input_b, np.array([target_gpu, 0.5]))

            # Принудительное исследование для B (ВСЕ 1000 тактов, плавно)
            action_b[0] = np.clip(action_b[0] + np.random.normal(0.0, 0.1), 0.0, 1.0)

            actuator.set_intensity(action_a[0], action_b[0])
            time.sleep(0.05)

            _, next_state = sensor.read()
            next_cpu = next_state.get("cpu_load", 0.0)
            next_gpu = next_state.get("gpu_load", 0.0)

            # Обучение на своих сенсорах
            next_input_a = np.array([next_cpu, last_msg_b])
            next_input_b = np.array([next_gpu, last_msg_a])
            
            brain_a.learn(input_a, action_a, next_input_a)
            brain_b.learn(input_b, action_b, next_input_b)

            # Обновляем сообщения
            last_msg_a = action_a[1]
            last_msg_b = action_b[1]

            energy.update(next_state)

            if step % 50 == 0:
                print(f"{step:3d} | {energy.energy:.3f} | {cpu:.2f} | {gpu:.2f} | {last_msg_a:.2f} | {last_msg_b:.2f}")

            if not energy.is_alive():
                print(f"*** Смерть на такте {step} ***")
                break

    finally:
        actuator.stop()
        print(f"Прожито тактов: {step}")

if __name__ == "__main__":
    main()
