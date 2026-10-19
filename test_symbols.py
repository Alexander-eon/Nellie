import time
import numpy as np
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
    cpu_in_zone, gpu_in_zone = 0, 0
    symbols_a = {0.0: 0, 0.5: 0, 1.0: 0}
    symbols_b = {0.0: 0, 0.5: 0, 1.0: 0}

    print("Шаг | Жизнь | Цель_CPU | CPU | Цель_GPU | GPU | Символ_A | Символ_B")
    print("-" * 80)

    step = 0
    try:
        for step in range(1000):
            t = step / 100.0
            target_cpu = 0.5 + 0.15 * np.sin(t)
            target_gpu = 0.5 + 0.15 * np.cos(t)
            
            # АСИММЕТРИЧНЫЙ ШОК: каждые 100 шагов резко меняем ТОЛЬКО ОДНУ цель
            if step % 100 == 0 and step > 0:
                if step % 200 == 0:
                    target_cpu = np.random.uniform(0.3, 0.7)
                else:
                    target_gpu = np.random.uniform(0.3, 0.7)

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

            # А кодирует изменение target_gpu (тренд) и шлет символ
            trend_gpu = target_gpu - (0.5 + 0.15 * np.cos((step-1)/100.0))
            trend_cpu = target_cpu - (0.5 + 0.15 * np.sin((step-1)/100.0))
            
            # Шлем 0.0 если падает, 1.0 если растет, 0.5 если стабильно
            if abs(trend_gpu) > 0.01:
                sym_a = 1.0 if trend_gpu > 0 else 0.0
            else:
                sym_a = 0.5

            if abs(trend_cpu) > 0.01:
                sym_b = 1.0 if trend_cpu > 0 else 0.0
            else:
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
            symbols_a[sym_a] = symbols_a.get(sym_a, 0) + 1
            symbols_b[sym_b] = symbols_b.get(sym_b, 0) + 1

            energy.update(next_state)

            if energy.cpu_min <= cpu <= energy.cpu_max: cpu_in_zone += 1
            if energy.gpu_min <= gpu <= energy.gpu_max: gpu_in_zone += 1

            if step % 50 == 0:
                print(f"{step:3d} | {energy.energy:.3f} | {target_cpu:.2f} | {cpu:.2f} | {target_gpu:.2f} | {gpu:.2f} |   {last_msg_a:.1f}    |   {last_msg_b:.1f}")

            if not energy.is_alive():
                print(f"*** Смерть на такте {step} ***")
                break

    finally:
        actuator.stop()
        print(f"\nПрожито тактов: {step + 1}")
        print(f"CPU в зоне: {cpu_in_zone} тактов ({cpu_in_zone/(step+1)*100:.1f}%)")
        print(f"GPU в зоне: {gpu_in_zone} тактов ({gpu_in_zone/(step+1)*100:.1f}%)")
        print(f"Агент А использовал символы: 0.0 -> {symbols_a[0.0]} раз, 0.5 -> {symbols_a[0.5]} раз, 1.0 -> {symbols_a[1.0]} раз")
        print(f"Агент B использовал символы: 0.0 -> {symbols_b[0.0]} раз, 0.5 -> {symbols_b[0.5]} раз, 1.0 -> {symbols_b[1.0]} раз")

if __name__ == "__main__":
    main()
