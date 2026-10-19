import time
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

def evaluate_pair(brains, steps=300):
    # brains: tuple of 6 brains (A_log, A_comm, B_log, B_comm, C_log, C_comm)
    a_log, a_comm, b_log, b_comm, c_log, c_comm = brains
    body = VirtualBody()
    energy = Energy(initial=1.0, drain_rate=0.003, bounty_rate=0.003, 
                    cpu_min=0.40, cpu_max=0.60, gpu_min=0.40, gpu_max=0.60)
    
    last_msg_a, last_msg_b = 0.5, 0.5
    msg_a_history, target_cpu_history = [], []
    
    for step in range(steps):
        t = step / 50.0 # Ускорили время для эволюции
        target_cpu = 0.5 + 0.15 * np.sin(t)
        target_gpu = 0.5 + 0.15 * np.cos(t)
        
        energy.cpu_min, energy.cpu_max = target_cpu - 0.05, target_cpu + 0.05
        energy.gpu_min, energy.gpu_max = target_gpu - 0.05, target_gpu + 0.05
        
        input_a = np.array([body.cpu, last_msg_b])
        input_b = np.array([body.gpu, last_msg_a])
        input_c = np.array([body.temp, last_msg_a])
        
        act_a_log = a_log.act(input_a, np.array([target_cpu]))
        act_b_log = b_log.act(input_b, np.array([target_gpu]))
        act_c_log = c_log.act(input_c, np.array([0.5]))
        
        act_a_comm = a_comm.act(input_a, np.array([target_cpu]))
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
        
        if energy.cpu_min <= body.cpu <= energy.cpu_max and energy.gpu_min <= body.gpu <= energy.gpu_max:
            energy.energy = min(1.0, energy.energy + 0.003)
        else:
            energy.energy -= 0.003
            
        if energy.energy <= 0:
            break
            
    # Fitness: корреляция msg_A с target_cpu (качество языка) * выживание
    corr = np.corrcoef(msg_a_history, target_cpu_history)[0, 1] if len(msg_a_history) > 10 else 0
    fitness = (step + 1) * max(0, corr) # Штраф за смерть и низкую корреляцию
    return fitness, corr, step + 1

def mutate(brain, sigma=0.05):
    new_brain = copy.deepcopy(brain)
    new_brain.W_s += np.random.normal(0, sigma, new_brain.W_s.shape)
    new_brain.W_a += np.random.normal(0, sigma, new_brain.W_a.shape)
    return new_brain

def main():
    pop_size = 10
    generations = 10
    mutation_sigma = 0.05
    
    # Инициализация популяции
    population = []
    for _ in range(pop_size):
        brains = (
            PredictiveBrain(2, 1), PredictiveBrain(2, 1),
            PredictiveBrain(2, 1), PredictiveBrain(2, 1),
            PredictiveBrain(2, 1), PredictiveBrain(2, 1)
        )
        brains[4].W_a[0, 0] = -0.5 # Врожденный рефлекс вентилятора
        population.append(brains)
        
    print("Поколение | Лучший Fitness | Корреляция msg_A | Выживание")
    print("-" * 65)
    
    for gen in range(generations):
        results = []
        for brains in population:
            fit, corr, life = evaluate_pair(brains)
            results.append((fit, corr, life, brains))
            
        # Сортировка по fitness
        results.sort(key=lambda x: x[0], reverse=True)
        
        best_fit, best_corr, best_life, best_brains = results[0]
        print(f"   {gen:2d}    |    {best_fit:6.1f}     |      {best_corr:.3f}       |    {best_life:3d}")
        
        # Отбор (топ-3) и размножение
        new_pop = []
        for i in range(pop_size):
            parent = results[i % 3][3] # Берем лучших
            child = tuple(mutate(b, mutation_sigma) for b in parent)
            new_pop.append(child)
            
        population = new_pop

if __name__ == "__main__":
    main()
