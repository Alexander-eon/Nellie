import time
import numpy as np
import csv
from predictive_brain import PredictiveBrain
from body_sensor import BodySensor
from body_actuator import BodyActuator
from energy import Energy

def main():
    sensor = BodySensor()
    actuator = BodyActuator()
    energy = Energy(initial=1.0, drain_rate=0.002, bounty_rate=0.003, 
                    cpu_min=0.40, cpu_max=0.60, gpu_min=0.40, gpu_max=0.60)

    brain_a = PredictiveBrain(n_sensors=2, n_actions=2, lr=0.05, lr_action=0.1)
    brain_b = PredictiveBrain(n_sensors=2, n_actions=2, lr=0.05, lr_action=0.1)

    actuator.start()
    time.sleep(1.0)

    current_target_cpu = 0.5
    current_target_gpu = 0.5
    last_msg_a = 0.5
    last_msg_b = 0.5
    
    cpu_in_zone, gpu_in_zone = 0, 0

    # Открываем файл для сохранения сырых данных
    with open('comm_data.csv', mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'target_cpu', 'target_gpu', 'cpu', 'gpu', 'msg_a', 'msg_b', 'act_a', 'act_b'])

        print("Шаг | Жизнь | Цель_CPU | CPU | Цель_GPU | GPU | msg_a | msg_b")
        print("-" * 75)

        step = 0
        try:
            for step in range(1000):
                if step % 100 == 0 and step > 0:
                    current_target_cpu = np.random.uniform(0.42, 0.58)
                    current_target_gpu = np.random.uniform(0.42, 0.58)

                _, state = sensor.read()
                cpu = state.get("cpu_load", 0.0)
                gpu = state.get("gpu_load", 0.0)

                input_a = np.array([cpu, last_msg_b])
                input_b = np.array([gpu, last_msg_a])

                target_a = np.array([last_msg_b, current_target_gpu]) 
                target_b = np.array([last_msg_a, current_target_cpu])

                action_a = brain_a.act(input_a, target_a)
                action_b = brain_b.act(input_b, target_b)

                action_a[0] = np.clip(action_a[0] + np.random.normal(0.0, 0.02), 0.0, 1.0)
                action_b[0] = np.clip(action_b[0] + np.random.normal(0.0, 0.02), 0.0, 1.0)

                actuator.set_intensity(action_a[0], action_b[0])
                time.sleep(0.05)

                _, next_state = sensor.read()
                next_cpu = next_state.get("cpu_load", 0.0)
                next_gpu = next_state.get("gpu_load", 0.0)

                next_msg_a = action_a[1]
                next_msg_b = action_b[1]

                next_input_a = np.array([next_cpu, next_msg_b])
                next_input_b = np.array([next_gpu, next_msg_a])

                brain_a.learn(input_a, action_a, next_input_a)
                brain_b.learn(input_b, action_b, next_input_b)

                last_msg_a = next_msg_a
                last_msg_b = next_msg_b

                energy.update(next_state)

                if 0.40 <= cpu <= 0.60: cpu_in_zone += 1
                if 0.40 <= gpu <= 0.60: gpu_in_zone += 1

                # Сохраняем строку в CSV
                writer.writerow([step, current_target_cpu, current_target_gpu, cpu, gpu, last_msg_a, last_msg_b, action_a[0], action_b[0]])

                if step % 50 == 0:
                    print(f"{step:3d} | {energy.energy:.3f} | {current_target_cpu:.2f} | {cpu:.2f} | {current_target_gpu:.2f} | {gpu:.2f} | {last_msg_a:.2f} | {last_msg_b:.2f}")

                if not energy.is_alive():
                    print(f"*** Смерть на такте {step} ***")
                    break

        finally:
            actuator.stop()
            print(f"\nПрожито тактов: {step + 1}")
            print(f"CPU в зоне: {cpu_in_zone} тактов ({cpu_in_zone/(step+1)*100:.1f}%)")
            print(f"GPU в зоне: {gpu_in_zone} тактов ({gpu_in_zone/(step+1)*100:.1f}%)")
            print("Данные сохранены в comm_data.csv")

if __name__ == "__main__":
    main()
