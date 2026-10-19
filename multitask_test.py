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

def evaluate_multitask(brains, steps=300):
    a_log, a_comm, b_log, b_comm, c_log, c_comm = brains
    body = VirtualBody()
    energy = Energy(initial=1.0, drain_rate=0.003, bounty_rate=0.003, 
                    cpu_min=0.40, cpu_max=0.60, gpu_min=0.40, gpu_max=0.60)
    last_msg_a, last_msg_b = 0.5, 0.5
    msg_a_history, target_cpu_history, target_gpu_history = [], [], []
    
    for step in range(steps):
        t = step / 50.0
        target_cpu = 0.5 + 0.15 * np.sin(t)
        target_gpu = 0.5 + 0.15 * np.cos(t)
        
        energy.cpu_min, energy.cpu_max = target_cpu - 0.05, target_cpu + 0.05
        energy.gpu_min, energy.gpu_max = target_gpu - 0.05, target_gpu + 0.05
        
        # Агент А хочет передать ДВЕ цели (CPU и GPU) через ОДИН канал
        # Мы даем ему две цели в target_a, но у него только один action_comm
        input_a = np.array([body.cpu, last_msg_b])
        input_b = np.array([body.gpu, last_msg_a])
        input_c = np.array([body.temp, last_msg_a])
        
        act_a_log = a_log.act(input_a, np.array([target_cpu]))
        act_b_log = b_log.act(input_b, np.array([target_gpu]))
        act_c_log = c_log.act(input_c, np.array([0.5]))
        
        # Агент А передает одно число, которое должно кодировать ОБЕ цели
        # Мы даем ему target = (target_cpu + target_gpu) / 2 (среднее)
        combined_target = (target_cpu + target_gpu) / 2.0
        act_a_comm = a_comm.act(input_a, np.array([combined_target]))
        
        act_b_comm = b_comm.act(input_b, np.array([target_gpu]))
        act_c_comm = c_comm.act(input_c, np.array([0.5]))
        
        act_a_log[0] = np.clip(act_a_log[0] + np.random.normal(0.0, 0.02), 0.0, 1.0)
        act_b_log[0] = np.clip(act_b_log[0] + np.random.normal(0.0, 0.02), 0.0, 1.0)
        act_c_log[0] = np.clip(act_c_log[0] + np.random.normal(0.0, 0.02), 0.0, 1.0)
        
        body.update(act_a_log[0], act_b_log[0], act_c_log[0])
        
        sym_a = np.clip(act_a_comm[0], 0.0, 1.0)
        sym_b = np.clip(act_b_comm[0], 0.0, 1.0)
        sym_c = np.clip(act_c_comm[0], 0.0, 1.0)
        
        next_input_a = np.array([body.cpu, sym_b])
        next_input_b = np.array([body.gpu, sym_a])
        next_input_c = np.array([body.temp, sym_a])
        
        a_log.learn(input_a, act_a_log, next_input_a)
        b_log.learn(input_b, act_b_log, next_input_b)
        c_log.learn(input_c, act_c_log, next_input_c)
        a_comm.learn(input_a, act_a_comm, next_input_a)
        b_comm.learn(input_b, act_b_comm, next_input_b)
        c_comm.learn(input_c, act_c_comm, next_input_c)
        
        last_msg_a, last_msg_b = sym_a, sym_b
        msg_a_history.append(sym_a)
        target_cpu_history.append(target_cpu)
        target_gpu_history.append(target_gpu)
        
        if energy.cpu_min <= body.cpu <= energy.cpu_max and energy.gpu_min <= body.gpu <= energy.gpu_max:
            energy.energy = min(1.0, energy.energy + 0.003)
        else:
            energy.energy -= 0.003
        if energy.energy <= 0:
            break
            
    # Корреляция с CPU и GPU
    corr_cpu = np.corrcoef(msg_a_history, target_cpu_history)[0, 1] if len(msg_a_history) > 10 else 0
    corr_gpu = np.corrcoef(msg_a_history, target_gpu_history)[0, 1] if len(msg_a_history) > 10 else 0
    # Fitness: среднее корреляций * выживание
    fitness = (step + 1) * max(0, (corr_cpu + corr_gpu) / 2)
    return fitness, corr_cpu, corr_gpu, step + 1, energy.energy

def create_population(size):
    pop = []
    for _ in range(size):
        brains = (
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
            fit, corr_cpu, corr_gpu, life, energy_left = evaluate_multitask(brains)
            results.append((fit, corr_cpu, corr_gpu, life, brains, energy_left))
        results.sort(key=lambda x: x[0], reverse=True)
        
        best = results[0]
        if best_overall is None or best[0] > best_overall[0]:
            best_overall = best
            
        print(f"  Поколение {gen}: Fitness={best[0]:.1f}, Корр CPU={best[1]:.3f}, Корр GPU={best[2]:.3f}, Жизнь={best[3]}")
        
        new_pop = []
        for i in range(len(population)):
            parent = results[i % 3][4]
            child = tuple(copy.deepcopy(b) for b in parent)
            for b in child:
                b.W_s += np.random.normal(0, 0.05, b.W_s.shape)
                b.W_a += np.random.normal(0, 0.05, b.W_a.shape)
            new_pop.append(child)
        population = new_pop
    return population, best_overall

def main():
    print("Эволюция в МНОГОЗАДАЧНОЙ среде (один канал для двух целей)...\n")
    pop = create_population(10)
    pop, best = evolve(pop, 20)
    
    print(f"\n=== ЛУЧШИЙ АГЕНТ (20 ПОКОЛЕНИЙ) ===")
    print(f"  Корреляция с CPU: {best[1]:.3f}")
    print(f"  Корреляция с GPU: {best[2]:.3f}")
    print(f"  Жизнь: {best[3]}")
    print(f"  Остаток энергии: {best[5]:.3f}")

if __name__ == "__main__":
    main()
