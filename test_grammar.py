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
        cooling = action_fan * 0.3 # Сделали охлаждение сильнее
        self.temp = np.clip(self.temp + heat - cooling, 0, 1)

def main():
    body = VirtualBody()
    energy = Energy(initial=1.0, drain_rate=0.003, bounty_rate=0.003, 
                    cpu_min=0.40, cpu_max=0.60, gpu_min=0.40, gpu_max=0.60)

    brain_a = PredictiveBrain(n_sensors=2, n_actions=2, lr=0.05, lr_action=0.1)
    brain_b = PredictiveBrain(n_sensors=2, n_actions=2, lr=0.05, lr_action=0.1)
    
    # C: 2 входа (Temp, msg_a), 1 выход (Fan)
    brain_c = PredictiveBrain(n_sensors=2, n_actions=1, lr=0.05, lr_action=0.1)
    # ВАЖНО: Даем C начальное знание, что вентилятор охлаждает (отрицательный вес)
    brain_c.W_a[0, 0] = -0.5 

    last_msg_a = 0.5
    last_msg_b = 0.5
    
    cpu_in_zone, gpu_in_zone, temp_in_zone = 0, 0, 0

    print("Шаг | Энергия | Цель_CPU | CPU | Цель_GPU | GPU | Temp | msg_A | act_Fan")
    print("-" * 85)

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
            
            target_a = np.array([target_cpu, 0.5])
            target_b = np.array([target_gpu, 0.5])
            
            action_a = brain_a.act(input_a, target_a)
            action_b = brain_b.act(input_b, target_b)

            # C видит температуру и символ от A
            input_c = np.array([body.temp, last_msg_a])
            target_c = np.array([target_temp])
            action_c = brain_c.act(input_c, target_c)

            action_a[0] = np.clip(action_a[0] + np.random.normal(0.0, 0.02), 0.0, 1.0)
            action_b[0] = np.clip(action_b[0] + np.random.normal(0.0, 0.02), 0.0, 1.0)
            action_c[0] = np.clip(action_c[0] + np.random.normal(0.0, 0.02), 0.0, 1.0)

            body.update(action_a[0], action_b[0], action_c[0])

            # Динамическое квантование (как в прошлый раз)
            err = abs(body.cpu - target_cpu)
            if err > 0.08:
                sym_a = 1.0 if body.cpu < target_cpu else 0.0
            else:
                sym_a = 0.5

            # Обучение (C учится на своих ошибках)
            next_input_a = np.array([body.cpu, 0.5])
            next_input_b = np.array([body.gpu, sym_a])
            next_input_c = np.array([body.temp, sym_a])

            brain_a.learn(input_a, action_a, next_input_a)
            brain_b.learn(input_b, action_b, next_input_b)
            brain_c.learn(input_c, action_c, next_input_c)

            last_msg_a = sym_a

            in_zone = (energy.cpu_min <= body.cpu <= energy.cpu_max) and (energy.gpu_min <= body.gpu <= energy.gpu_max)
            if in_zone:
                energy.energy = min(1.0, energy.energy + 0.003)
            else:
                energy.energy -= 0.003

            if energy.cpu_min <= body.cpu <= energy.cpu_max: cpu_in_zone += 1
            if energy.gpu_min <= body.gpu <= energy.gpu_max: gpu_in_zone += 1
            if 0.4 <= body.temp <= 0.6: temp_in_zone += 1

            if step % 50 == 0:
                print(f"{step:3d} | {energy.energy:.3f} | {target_cpu:.2f} | {body.cpu:.2f} | {target_gpu:.2f} | {body.gpu:.2f} | {body.temp:.2f} | {sym_a:.1f} | {action_c[0]:.2f}")

            if energy.energy <= 0:
                print(f"*** Смерть на такте {step} ***")
                break

    finally:
        print(f"\nПрожито тактов: {step + 1}")
        print(f"CPU в зоне: {cpu_in_zone/(step+1)*100:.1f}%")
        print(f"GPU в зоне: {gpu_in_zone/(step+1)*100:.1f}%")
        print(f"Temp в зоне: {temp_in_zone/(step+1)*100:.1f}%")

if __name__ == "__main__":
    main()
