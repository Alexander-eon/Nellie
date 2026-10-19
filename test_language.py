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
                    cpu_min=0.40, cpu_max=0.60, gpu_min=0.40, gpu_max=0.60)

    # Теперь у нас 4 сенсора (свой сенсор + 3 сообщения) и 4 действия (свое действие + 3 сообщения)
    brain_a = PredictiveBrain(n_sensors=4, n_actions=4, lr=0.05, lr_action=0.1)
    brain_b = PredictiveBrain(n_sensors=4, n_actions=4, lr=0.05, lr_action=0.1)

    actuator.start()
    time.sleep(1.0)

    last_msg_a = np.array([0.5, 0.5, 0.5])
    last_msg_b = np.array([0.5, 0.5, 0.5])
    
    cpu_in_zone, gpu_in_zone = 0, 0

    print("Шаг | Жизнь | Цель_CPU | CPU | Цель_GPU | GPU | msg_A (цель/тренд/энергия)")
    print("-" * 85)

    step = 0
    try:
        for step in range(1000):
            # Синусоидальные цели (плавные)
            t = step / 100.0
            target_cpu = 0.5 + 0.15 * np.sin(t)
            target_gpu = 0.5 + 0.15 * np.cos(t)
            
            # Динамическая зона (следует за целью)
            energy.cpu_min = target_cpu - 0.05
            energy.cpu_max = target_cpu + 0.05
            energy.gpu_min = target_gpu - 0.05
            energy.gpu_max = target_gpu + 0.05

            _, state = sensor.read()
            cpu = state.get("cpu_load", 0.0)
            gpu = state.get("gpu_load", 0.0)

            # Входы: А видит свой CPU и 3 сообщения от B
            input_a = np.array([cpu, last_msg_b[0], last_msg_b[1], last_msg_b[2]])
            # Входы: B видит свой GPU и 3 сообщения от A
            input_b = np.array([gpu, last_msg_a[0], last_msg_a[1], last_msg_a[2]])

            # Цели для действий А: [цель_CPU_из_сообщения, цель_GPU, тренд_GPU, энергия_A]
            # А кодирует в свои сообщения: target_gpu, тренд, и свою энергию (как алярм)
            trend_gpu = np.cos(t) - np.cos((step-1)/100.0) # производная
            target_a = np.array([last_msg_b[0], target_gpu, trend_gpu, energy.energy])
            
            # Цели для действий B: [цель_GPU_из_сообщения, цель_CPU, тренд_CPU, энергия_B]
            trend_cpu = np.sin(t) - np.sin((step-1)/100.0)
            target_b = np.array([last_msg_a[0], target_cpu, trend_cpu, energy.energy])

            action_a = brain_a.act(input_a, target_a)
            action_b = brain_b.act(input_b, target_b)

            # Шум только на физическое действие
            action_a[0] = np.clip(action_a[0] + np.random.normal(0.0, 0.02), 0.0, 1.0)
            action_b[0] = np.clip(action_b[0] + np.random.normal(0.0, 0.02), 0.0, 1.0)

            actuator.set_intensity(action_a[0], action_b[0])
            time.sleep(0.05)

            _, next_state = sensor.read()
            next_cpu = next_state.get("cpu_load", 0.0)
            next_gpu = next_state.get("gpu_load", 0.0)

            # Новые сообщения
            next_msg_a = action_a[1:]
            next_msg_b = action_b[1:]

            next_input_a = np.array([next_cpu, next_msg_b[0], next_msg_b[1], next_msg_b[2]])
            next_input_b = np.array([next_gpu, next_msg_a[0], next_msg_a[1], next_msg_a[2]])

            brain_a.learn(input_a, action_a, next_input_a)
            brain_b.learn(input_b, action_b, next_input_b)

            last_msg_a = next_msg_a
            last_msg_b = next_msg_b

            energy.update(next_state)

            if energy.cpu_min <= cpu <= energy.cpu_max: cpu_in_zone += 1
            if energy.gpu_min <= gpu <= energy.gpu_max: gpu_in_zone += 1

            if step % 50 == 0:
                print(f"{step:3d} | {energy.energy:.3f} | {target_cpu:.2f} | {cpu:.2f} | {target_gpu:.2f} | {gpu:.2f} | {last_msg_a[0]:.2f}/{last_msg_a[1]:.2f}/{last_msg_a[2]:.2f}")

            if not energy.is_alive():
                print(f"*** Смерть на такте {step} ***")
                break

    finally:
        actuator.stop()
        print(f"\nПрожито тактов: {step + 1}")
        print(f"CPU в зоне: {cpu_in_zone} тактов ({cpu_in_zone/(step+1)*100:.1f}%)")
        print(f"GPU в зоне: {gpu_in_zone} тактов ({gpu_in_zone/(step+1)*100:.1f}%)")

if __name__ == "__main__":
    main()
