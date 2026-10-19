import time
import numpy as np
import pandas as pd
from predictive_brain import PredictiveBrain
from body_sensor import BodySensor
from body_actuator import BodyActuator
from energy import Energy

def main():
    sensor = BodySensor()
    actuator = BodyActuator()
    energy = Energy(initial=1.0, drain_rate=0.004, bounty_rate=0.003,
                    cpu_min=0.40, cpu_max=0.60, gpu_min=0.40, gpu_max=0.60)

    brain_a = PredictiveBrain(n_sensors=2, n_actions=2, lr=0.05, lr_action=0.1)
    brain_b = PredictiveBrain(n_sensors=2, n_actions=2, lr=0.05, lr_action=0.1)

    actuator.start()
    time.sleep(1.0)

    last_msg_a = 0.5
    last_msg_b = 0.5
    data = []

    step = 0
    try:
        for step in range(1000):
            t = step / 100.0
            target_cpu = 0.5 + 0.15 * np.sin(t)
            target_gpu = 0.5 + 0.15 * np.cos(t)
            
            event = "stable"
            # ШОК: для CPU на 100-м шаге, для GPU на 105-м (задержка 5 тактов)
            if step == 100:
                target_cpu = 0.7 # Резкий скачок вверх
                event = "SHOCK_CPU"
            elif step == 105:
                target_gpu = 0.7 # Резкий скачок вверх (через 5 тактов)
                event = "SHOCK_GPU"

            energy.cpu_min = target_cpu - 0.05
            energy.cpu_max = target_cpu + 0.05
            energy.gpu_min = target_gpu - 0.05
            energy.gpu_max = target_gpu + 0.05

            _, state = sensor.read()
            cpu = state.get("cpu_load", 0.0)
            gpu = state.get("gpu_load", 0.0)

            input_a = np.array([cpu, last_msg_b])
            input_b = np.array([gpu, last_msg_a])

            boost_a = (last_msg_b - 0.5) * 0.4
            boost_b = (last_msg_a - 0.5) * 0.4

            target_a = np.array([target_cpu + boost_a, 0.5])
            target_b = np.array([target_gpu + boost_b, 0.5])

            action_a = brain_a.act(input_a, target_a)
            action_b = brain_b.act(input_b, target_b)

            # Агент А кодирует изменение CPU
            sym_a = 1.0 if (step >= 100 and step < 110) else 0.5
            # Агент B должен предсказать шок GPU, услышав от A
            sym_b = 0.5

            action_a[0] = np.clip(action_a[0] + np.random.normal(0.0, 0.02), 0.0, 1.0)
            action_b[0] = np.clip(action_b[0] + np.random.normal(0.0, 0.02), 0.0, 1.0)

            actuator.set_intensity(action_a[0], action_b[0])
            time.sleep(0.05)

            _, next_state = sensor.read()
            next_cpu = next_state.get("cpu_load", 0.0)
            next_gpu = next_state.get("gpu_load", 0.0)

            next_input_a = np.array([next_cpu, sym_b])
            next_input_b = np.array([next_gpu, sym_a])

            brain_a.learn(input_a, action_a, next_input_a)
            brain_b.learn(input_b, action_b, next_input_b)

            last_msg_a = sym_a
            last_msg_b = sym_b

            energy.update(next_state)
            data.append([step, target_cpu, target_gpu, cpu, gpu, sym_a, sym_b, event])

            if step % 10 == 0 and 90 <= step <= 120:
                print(f"Шаг {step}: CPU={cpu:.2f} (цель {target_cpu:.2f}), GPU={gpu:.2f} (цель {target_gpu:.2f}), sym_a={sym_a:.1f}, sym_b={sym_b:.1f}")

            if not energy.is_alive():
                print(f"*** Смерть на такте {step} ***")
                break

    finally:
        actuator.stop()
        df = pd.DataFrame(data, columns=['step', 'target_cpu', 'target_gpu', 'cpu', 'gpu', 'sym_a', 'sym_b', 'event'])
        df.to_csv('symbols_data.csv', index=False)
        print(f"\nПрожито тактов: {step+1}")

if __name__ == "__main__":
    main()
