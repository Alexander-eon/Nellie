import time
import numpy as np
import pandas as pd
from predictive_brain import PredictiveBrain
from body_sensor import BodySensor
from body_actuator import BodyActuator
from energy import Energy
import analyze_te # Импортируем нашу функцию TE

def run_experiment(seed):
    np.random.seed(seed)
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

    data = []

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

        data.append([step, current_target_cpu, current_target_gpu, cpu, gpu, last_msg_a, last_msg_b, action_a[0], action_b[0]])

    actuator.stop()
    
    df = pd.DataFrame(data, columns=['step', 'target_cpu', 'target_gpu', 'cpu', 'gpu', 'msg_a', 'msg_b', 'act_a', 'act_b'])
    df.to_csv(f'comm_data_seed_{seed}.csv', index=False)
    
    # Считаем TE
    te_ma_gpu = analyze_te.transfer_entropy(df['msg_a'].values, df['gpu'].values)
    te_mb_cpu = analyze_te.transfer_entropy(df['msg_b'].values, df['cpu'].values)
    
    return cpu_in_zone, gpu_in_zone, te_ma_gpu, te_mb_cpu

if __name__ == "__main__":
    print("Запуск 3 прогонов для проверки воспроизводимости...")
    results = []
    for seed in [1, 2, 3]:
        print(f" -> Запуск seed={seed} (займет ~50 секунд)...")
        cpu_z, gpu_z, te_a, te_b = run_experiment(seed)
        results.append((seed, cpu_z, gpu_z, te_a, te_b))
        
    print("\n=== ИТОГИ ВОСПРОИЗВОДИМОСТИ ===")
    print("Seed | CPU в зоне | GPU в зоне | TE(msg_a->GPU) | TE(msg_b->CPU)")
    print("-" * 65)
    for r in results:
        print(f" {r[0]}   |    {r[1]}/1000   |    {r[2]}/1000   |     {r[3]:.4f}     |     {r[4]:.4f}")
