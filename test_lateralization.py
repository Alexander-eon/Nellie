import time
import numpy as np
from predictive_brain import PredictiveBrain
from energy import Energy

class VirtualBody:
    def __init__(self):
        self.cpu = 0.1
        self.gpu = 0.1
        self.temp = 0.5
        
    def update(self, action_cpu, action_gpu, action_fan):
        self.cpu = np.clip(self.cpu + (action_cpu - self.cpu) * 0.1, 0, 1)
        self.gpu = np.clip(self.gpu + (action_gpu - self.gpu) * 0.1, 0, 1)
        heat = (self.cpu + self.gpu) * 0.15
        cooling = action_fan * 0.3
        self.temp = np.clip(self.temp + heat - cooling, 0, 1)

def main():
    body = VirtualBody()
    energy = Energy(initial=1.0, drain_rate=0.003, bounty_rate=0.003, 
                    cpu_min=0.40, cpu_max=0.60, gpu_min=0.40, gpu_max=0.60)

    # Полушария
    brain_a_logic = PredictiveBrain(n_sensors=2, n_actions=1, lr=0.05, lr_action=0.1)
    brain_a_comm = PredictiveBrain(n_sensors=2, n_actions=1, lr=0.05, lr_action=0.1)
    
    brain_b_logic = PredictiveBrain(n_sensors=2, n_actions=1, lr=0.05, lr_action=0.1)
    brain_b_comm = PredictiveBrain(n_sensors=2, n_actions=1, lr=0.05, lr_action=0.1)
    
    brain_c_logic = PredictiveBrain(n_sensors=2, n_actions=1, lr=0.05, lr_action=0.1)
    brain_c_comm = PredictiveBrain(n_sensors=2, n_actions=1, lr=0.05, lr_action=0.1)

    brain_c_logic.W_a[0, 0] = -0.5

    last_msg_a = 0.5
    last_msg_b = 0.5
    last_msg_c = 0.5
    
    cpu_in_zone, gpu_in_zone, temp_in_zone = 0, 0, 0
    msg_history = []

    print("Шаг | Энергия | Цель_CPU | CPU | Цель_GPU | GPU | Temp | msg_A | msg_B | msg_C")
    print("-" * 95)

    step = 0
    try:
        for step in range(1000):
            t = step / 100.0
            target_cpu = 0.5 + 0.15 * np.sin(t)
            target_gpu = 0.5 + 0.15 * np.cos(t)
            target_temp = 0.5

            energy.cpu_min, energy.cpu_max = target_cpu - 0.05, target_cpu + 0.05
            energy.gpu_min, energy.gpu_max = target_gpu - 0.05, target_gpu + 0.05

            input_a = np.array([body.cpu, last_msg_b])
            input_b = np.array([body.gpu, last_msg_a])
            input_c = np.array([body.temp, last_msg_a])

            # Логика: агенты держат СВОИ цели
            action_a_log = brain_a_logic.act(input_a, np.array([target_cpu]))
            action_b_log = brain_b_logic.act(input_b, np.array([target_gpu]))
            action_c_log = brain_c_logic.act(input_c, np.array([target_temp]))

            # КОММУНИКАЦИЯ: агенты передают СВОИ цели
            # Агент А хочет передать target_cpu (свою цель)
            action_a_comm = brain_a_comm.act(input_a, np.array([target_cpu]))
            
            # Агент В хочет передать target_gpu (свою цель)
            action_b_comm = brain_b_comm.act(input_b, np.array([target_gpu]))
            
            # Агент С хочет передать target_temp (свою цель)
            action_c_comm = brain_c_comm.act(input_c, np.array([target_temp]))

            action_a_log[0] = np.clip(action_a_log[0] + np.random.normal(0.0, 0.02), 0.0, 1.0)
            action_b_log[0] = np.clip(action_b_log[0] + np.random.normal(0.0, 0.02), 0.0, 1.0)
            action_c_log[0] = np.clip(action_c_log[0] + np.random.normal(0.0, 0.02), 0.0, 1.0)

            body.update(action_a_log[0], action_b_log[0], action_c_log[0])

            sym_a = np.clip(action_a_comm[0], 0.0, 1.0)
            sym_b = np.clip(action_b_comm[0], 0.0, 1.0)
            sym_c = np.clip(action_c_comm[0], 0.0, 1.0)

            next_input_a = np.array([body.cpu, sym_b])
            next_input_b = np.array([body.gpu, sym_a])
            next_input_c = np.array([body.temp, sym_a])

            brain_a_logic.learn(input_a, action_a_log, next_input_a)
            brain_b_logic.learn(input_b, action_b_log, next_input_b)
            brain_c_logic.learn(input_c, action_c_log, next_input_c)

            brain_a_comm.learn(input_a, action_a_comm, next_input_a)
            brain_b_comm.learn(input_b, action_b_comm, next_input_b)
            brain_c_comm.learn(input_c, action_c_comm, next_input_c)

            last_msg_a = sym_a
            last_msg_b = sym_b
            last_msg_c = sym_c
            msg_history.append([sym_a, sym_b, sym_c])

            in_zone = (energy.cpu_min <= body.cpu <= energy.cpu_max) and (energy.gpu_min <= body.gpu <= energy.gpu_max)
            if in_zone:
                energy.energy = min(1.0, energy.energy + 0.003)
            else:
                energy.energy -= 0.003

            if energy.cpu_min <= body.cpu <= energy.cpu_max: cpu_in_zone += 1
            if energy.gpu_min <= body.gpu <= energy.gpu_max: gpu_in_zone += 1
            if 0.4 <= body.temp <= 0.6: temp_in_zone += 1

            if step % 50 == 0:
                print(f"{step:3d} | {energy.energy:.3f} | {target_cpu:.2f} | {body.cpu:.2f} | {target_gpu:.2f} | {body.gpu:.2f} | {body.temp:.2f} | {sym_a:.2f} | {sym_b:.2f} | {sym_c:.2f}")

            if energy.energy <= 0:
                print(f"*** Смерть на такте {step} ***")
                break

    finally:
        print(f"\nПрожито тактов: {step + 1}")
        print(f"CPU в зоне: {cpu_in_zone/(step+1)*100:.1f}%")
        print(f"GPU в зоне: {gpu_in_zone/(step+1)*100:.1f}%")
        print(f"Temp в зоне: {temp_in_zone/(step+1)*100:.1f}%")
        
        msg_arr = np.array(msg_history)
        if len(msg_arr) > 0:
            unique_levels = len(np.unique(np.round(msg_arr, 2)))
            print(f"Уникальных уровней сигнала (округление до 0.01): {unique_levels}")
            
            # Проверяем, передает ли Агент А свою цель
            corr_target_sym = np.corrcoef(msg_arr[:, 0], [0.5 + 0.15 * np.sin(s/100.0) for s in range(len(msg_arr))])[0, 1]
            print(f"Корреляция msg_A с target_cpu: {corr_target_sym:.3f}")
            print(f"Примеры msg_A за последние 10 шагов: {[round(m[0], 2) for m in msg_history[-10:]]}")

if __name__ == "__main__":
    main()
