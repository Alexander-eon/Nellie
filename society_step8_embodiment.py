import numpy as np
import time
import psutil
import multiprocessing
from predictive_brain import PredictiveBrain

# Постоянный фоновый процесс (грузит одно ядро)
def cpu_stress_loop(shared_action, stop_flag):
    while not stop_flag.value:
        action = shared_action.value
        if action > 0.1:
            work_time = action * 0.01
            sleep_time = (1.0 - action) * 0.01
            end_work = time.time() + work_time
            while time.time() < end_work:
                pass
            time.sleep(sleep_time)
        else:
            time.sleep(0.01)

class RealBody:
    def __init__(self):
        self.cpu = 0.5
        self.temp = 0.5
        self.fatigue = 0.0
        
        self.shared_action = multiprocessing.Value('d', 0.0)
        self.stop_flag = multiprocessing.Value('b', False)
        
        # ЗАПУСКАЕМ ПРОЦЕССЫ НА ВСЕХ ЯДРАХ
        num_cores = multiprocessing.cpu_count()
        self.stress_processes = []
        for _ in range(num_cores):
            p = multiprocessing.Process(target=cpu_stress_loop, args=(self.shared_action, self.stop_flag))
            p.start()
            self.stress_processes.append(p)
        
    def update(self, action_cpu):
        self.shared_action.value = action_cpu
        self.cpu = psutil.cpu_percent(interval=0.05) / 100.0
        
        target_temp = 0.3 + (self.cpu * 0.6)
        self.temp = self.temp + (target_temp - self.temp) * 0.1
        
        load = self.cpu
        self.fatigue = np.clip(self.fatigue + (load - 0.5) * 0.02, 0, 1)
        
    def stop(self):
        self.stop_flag.value = True
        for p in self.stress_processes:
            p.join()

def evaluate_embodiment(brains, steps=100):
    a_log, a_comm, b_log, b_comm, c_log, c_comm, d_log, d_comm = brains
    a_log.lr_action = 0.1 
    
    body = RealBody()
    energy_a = 1.0
    last_msg = 0.5
    
    print("\n=== ЗАПУСК В РЕАЛЬНОМ МИРЕ (МНОГОЯДЕРНАЯ НАГРУЗКА) ===")
    print("Шаг | Реальный CPU | Вирт. Temp | Действие_A | Энергия_A")
    
    for step in range(steps):
        target_cpu = 0.5
        
        input_a = np.array([body.cpu, body.temp, body.fatigue, energy_a, last_msg])
        act_a = a_log.act(input_a, np.array([target_cpu]))
        
        body.update(act_a[0])
        
        if step % 5 == 0:
            print(f"{step:3d} |    {body.cpu:.2f}      |    {body.temp:.2f}    |    {act_a[0]:.2f}    |   {energy_a:.2f}")
        
        # Успех, если в зоне
        if 0.3 <= body.cpu <= 0.7 and 0.3 <= body.temp <= 0.7:
            energy_a = min(1.0, energy_a + 0.02)
        else:
            energy_a -= 0.01
            
        next_input_a = np.array([body.cpu, body.temp, body.fatigue, energy_a, act_a[0]])
        a_log.learn(input_a, act_a, next_input_a)
            
        if energy_a <= 0:
            print(f"Агент А умер на шаге {step} от истощения!")
            break
            
        time.sleep(0.05)
        
    body.stop()
    return step + 1, energy_a

def main():
    print("ЭТАП 8: Воплощение (Управление реальным процессором, все ядра)...")
    
    brains = (PredictiveBrain(5, 1), PredictiveBrain(5, 1),
              PredictiveBrain(4, 1), PredictiveBrain(4, 1),
              PredictiveBrain(4, 1), PredictiveBrain(4, 1),
              PredictiveBrain(4, 1), PredictiveBrain(4, 1))
    
    life, final_energy = evaluate_embodiment(brains, steps=100)
    print(f"\nПрожито тактов: {life}")
    print(f"Остаток энергии: {final_energy:.3f}")

if __name__ == "__main__":
    main()
