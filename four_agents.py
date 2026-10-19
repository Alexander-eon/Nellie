import numpy as np
import copy
from predictive_brain import PredictiveBrain
from energy import Energy

class VirtualBody:
    def __init__(self):
        self.cpu, self.gpu, self.temp = 0.1, 0.1, 0.5
    def update(self, action_cpu, action_gpu, action_fan):
        self.cpu = np.clip(self.cpu + (action_cpu - self.cpu) * 0.1, 0, 1)
        self.gpu = np.clip(self.gpu + (action_gpu - self.gpu) * 0.1, 0, 1)
        heat = (self.cpu + self.gpu) * 0.15
        cooling = action_fan * 0.3
        self.temp = np.clip(self.temp + heat - cooling, 0, 1)

def evaluate_four(brains, steps=300):
    # Распаковываем 8 мозгов (4 агента * 2 полушария)
    a_log, a_comm, b_log, b_comm, c_log, c_comm, d_log, d_comm = brains
    body = VirtualBody()
    energy = Energy(initial=1.0, drain_rate=0.003, bounty_rate=0.003, 
                    cpu_min=0.40, cpu_max=0.60, gpu_min=0.40, gpu_max=0.60)
    
    last_msg_a, last_msg_b, last_msg_c, last_msg_d = 0.5, 0.5, 0.5, 0.5
    msg_a_history, target_cpu_history = [], []
    
    for step in range(steps):
        t = step / 50.0
        target_cpu = 0.5 + 0.15 * np.sin(t)
        target_gpu = 0.5 + 0.15 * np.cos(t)
        
        # Каждые 50 шагов — кризис (энергия падает)
        crisis = False
        if step % 50 == 0 and step > 0:
            energy.energy -= 0.2
            crisis = True
        
        energy.cpu_min, energy.cpu_max = target_cpu - 0.05, target_cpu + 0.05
        energy.gpu_min, energy.gpu_max = target_gpu - 0.05, target_gpu + 0.05
        
        # Входы: каждый агент видит свой сенсор и сообщение от предыдущего
        input_a = np.array([body.cpu, last_msg_b])
        input_b = np.array([body.gpu, last_msg_a])
        input_c = np.array([body.temp, last_msg_a])
        input_d = np.array([energy.energy, last_msg_a]) # D видит энергию
        
        # Логика
        act_a_log = a_log.act(input_a, np.array([target_cpu]))
        act_b_log = b_log.act(input_b, np.array([target_gpu]))
        act_c_log = c_log.act(input_c, np.array([0.5]))
        act_d_log = d_log.act(input_d, np.array([0.5])) # D хочет держать энергию 0.5
        
        # Коммуникация
        act_a_comm = a_comm.act(input_a, np.array([target_cpu]))
        act_b_comm = b_comm.act(input_b, np.array([target_gpu]))
        act_c_comm = c_comm.act(input_c, np.array([0.5]))
        # D передает сигнал тревоги, если энергия < 0.4
        comm_target_d = 0.0 if energy.energy < 0.4 else 0.5
        act_d_comm = d_comm.act(input_d, np.array([comm_target_d]))
        
        act_a_log[0] = np.clip(act_a_log[0] + np.random.normal(0.0, 0.02), 0.0, 1.0)
        act_b_log[0] = np.clip(act_b_log[0] + np.random.normal(0.0, 0.02), 0.0, 1.0)
        act_c_log[0] = np.clip(act_c_log[0] + np.random.normal(0.0, 0.02), 0.0, 1.0)
        act_d_log[0] = np.clip(act_d_log[0] + np.random.normal(0.0, 0.02), 0.0, 1.0)
        
        body.update(act_a_log[0], act_b_log[0], act_c_log[0])
        
        sym_a = np.clip(act_a_comm[0], 0.0, 1.0)
        sym_b = np.clip(act_b_comm[0], 0.0, 1.0)
        sym_c = np.clip(act_c_comm[0], 0.0, 1.0)
        sym_d = np.clip(act_d_comm[0], 0.0, 1.0)
        
        next_input_a = np.array([body.cpu, sym_b])
        next_input_b = np.array([body.gpu, sym_a])
        next_input_c = np.array([body.temp, sym_a])
        next_input_d = np.array([energy.energy, sym_a])
        
        a_log.learn(input_a, act_a_log, next_input_a)
        b_log.learn(input_b, act_b_log, next_input_b)
        c_log.learn(input_c, act_c_log, next_input_c)
        d_log.learn(input_d, act_d_log, next_input_d)
        
        a_comm.learn(input_a, act_a_comm, next_input_a)
        b_comm.learn(input_b, act_b_comm, next_input_b)
        c_comm.learn(input_c, act_c_comm, next_input_c)
        d_comm.learn(input_d, act_d_comm, next_input_d)
        
        last_msg_a, last_msg_b, last_msg_c, last_msg_d = sym_a, sym_b, sym_c, sym_d
        msg_a_history.append(sym_a)
        target_cpu_history.append(target_cpu)
        
        if energy.cpu_min <= body.cpu <= energy.cpu_max and energy.gpu_min <= body.gpu <= energy.gpu_max:
            energy.energy = min(1.0, energy.energy + 0.003)
        else:
            energy.energy -= 0.003
        if energy.energy <= 0:
            break
            
    corr = np.corrcoef(msg_a_history, target_cpu_history)[0, 1] if len(msg_a_history) > 10 else 0
    fitness = (step + 1) * max(0, corr)
    return fitness, corr, step + 1, energy.energy

def create_population(size):
    pop = []
    for _ in range(size):
        brains = (
            PredictiveBrain(2, 1), PredictiveBrain(2, 1),
            PredictiveBrain(2, 1), PredictiveBrain(2, 1),
            PredictiveBrain(2, 1), PredictiveBrain(2, 1),
            PredictiveBrain(2, 1), PredictiveBrain(2, 1)
        )
        brains[4].W_a[0, 0] = -0.5
        pop.append(brains)
    return pop

def evolve(population, generations):
    best_overall = None
    for gen in range(generations):
        results = []
        for brains in population:
            fit, corr, life, energy_left = evaluate_four(brains)
            results.append((fit, corr, life, brains, energy_left))
        results.sort(key=lambda x: x[0], reverse=True)
        
        best = results[0]
        if best_overall is None or best[0] > best_overall[0]:
            best_overall = best
            
        print(f"  Поколение {gen}: Fitness={best[0]:.1f}, Корр={best[1]:.3f}, Жизнь={best[2]}, Энергия={best[4]:.3f}")
        
        new_pop = []
        for i in range(len(population)):
            parent = results[i % 3][3]
            child = tuple(copy.deepcopy(b) for b in parent)
            for b in child:
                b.W_s += np.random.normal(0, 0.05, b.W_s.shape)
                b.W_a += np.random.normal(0, 0.05, b.W_a.shape)
            new_pop.append(child)
        population = new_pop
    return population, best_overall

def main():
    print("Эволюция с ЧЕТЫРЬМЯ агентами (A: CPU, B: GPU, C: Temp, D: Энергия)...\n")
    pop = create_population(10)
    pop, best = evolve(pop, 20)
    
    print(f"\n=== ЛУЧШИЙ АГЕНТ (20 ПОКОЛЕНИЙ) ===")
    print(f"  Корреляция: {best[1]:.3f}")
    print(f"  Жизнь: {best[2]}")
    print(f"  Остаток энергии: {best[4]:.3f}")

if __name__ == "__main__":
    main()
