import time
import numpy as np
from predictive_brain import PredictiveBrain
from body_sensor import BodySensor
from body_actuator import BodyActuator
from energy import Energy

def run_scenario(sever_channel=False):
    np.random.seed(42)
    sensor = BodySensor()
    actuator = BodyActuator()
    energy = Energy(initial=1.0, drain_rate=0.003, bounty_rate=0.003, 
                    cpu_min=0.45, cpu_max=0.55, gpu_min=0.45, gpu_max=0.55)

    brain_a = PredictiveBrain(n_sensors=2, n_actions=2, lr=0.05, lr_action=0.1)
    brain_b = PredictiveBrain(n_sensors=2, n_actions=2, lr=0.05, lr_action=0.1)

    actuator.start()
    time.sleep(1.0)

    current_target_cpu = 0.5
    current_target_gpu = 0.5
    last_msg_a = 0.5
    last_msg_b = 0.5
    cpu_in_zone, gpu_in_zone = 0, 0
    steps_survived = 0

    for step in range(1000):
        if step % 100 == 0 and step > 0:
            # РЕАЛИСТИЧНЫЕ ЦЕЛИ: от 0.3 до 0.7 (сервер это выдержит)
            current_target_cpu = np.random.uniform(0.3, 0.7)
            current_target_gpu = np.random.uniform(0.3, 0.7)
            
            # УЗКАЯ ЗОНА: +/- 0.04
            energy.cpu_min = current_target_cpu - 0.04
            energy.cpu_max = current_target_cpu + 0.04
            energy.gpu_min = current_target_gpu - 0.04
            energy.gpu_max = current_target_gpu + 0.04

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

        if sever_channel:
            next_msg_a, next_msg_b = 0.5, 0.5
        else:
            next_msg_a, next_msg_b = action_a[1], action_b[1]

        next_input_a = np.array([next_cpu, next_msg_b])
        next_input_b = np.array([next_gpu, next_msg_a])

        brain_a.learn(input_a, action_a, next_input_a)
        brain_b.learn(input_b, action_b, next_input_b)

        last_msg_a, last_msg_b = next_msg_a, next_msg_b
        energy.update(next_state)

        if energy.cpu_min <= cpu <= energy.cpu_max: cpu_in_zone += 1
        if energy.gpu_min <= gpu <= energy.gpu_max: gpu_in_zone += 1

        steps_survived = step + 1
        if not energy.is_alive():
            break

    actuator.stop()
    return steps_survived, cpu_in_zone, gpu_in_zone

if __name__ == "__main__":
    print("Сценарий 1: С ВКЛЮЧЕННЫМ каналом связи...")
    s_norm, c_norm, g_norm = run_scenario(sever_channel=False)
    print(f"  -> Жизнь: {s_norm} | CPU в зоне: {c_norm} ({c_norm/s_norm*100:.1f}%) | GPU в зоне: {g_norm} ({g_norm/s_norm*100:.1f}%)")
    
    print("\nСценарий 2: С РАЗОРВАННЫМ каналом связи (глухие агенты)...")
    s_sev, c_sev, g_sev = run_scenario(sever_channel=True)
    print(f"  -> Жизнь: {s_sev} | CPU в зоне: {c_sev} ({c_sev/s_sev*100:.1f}%) | GPU в зоне: {g_sev} ({g_sev/s_sev*100:.1f}%)")
    
    print("\n=== ВЕРДИКТ ===")
    # Если без связи умерли раньше или провели в зоне значительно меньше времени
    if s_sev < s_norm * 0.8 or c_sev < c_norm * 0.6:
        print("ЭФФЕКТ ДОКАЗАН! Коммуникация критически важна для выживания.")
    else:
        print("Агенты всё ещё справляются. Надо усложнить.")
