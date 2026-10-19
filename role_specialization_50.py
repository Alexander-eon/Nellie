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

def evaluate_specialization(brains, steps=300):
    a_log, a_comm, b_log, b_comm, c_log, c_comm, d_log, d_comm = brains
    body = VirtualBody()
    energy = Energy(initial=1.0, drain_rate=0.003, bounty_rate=0.003, 
                    cpu_min=0.40, cpu_max=0.60, gpu_min=0.40, gpu_max=0.60)
    last_msgs = [0.5, 0.5, 0.5, 0.5]
    
    for step in range(steps):
        t = step / 50.0
        target_cpu = 0.5 + 0.15 * np.sin(t)
        target_gpu = 0.5 + 0.15 * np.cos(t)
        
        # Кризис: потеря 0.05 каждые 50 шагов
        if step % 50 == 0 and step > 0:
            energy.energy -= 0.05
        
        input_a = np.array([body.cpu, last_msgs[1]])
        input_b = np.array([body.gpu, last_msgs[2]])
        input_c = np.array([body.temp, last_msgs[3]])
        input_d = np.array([energy.energy, last_msgs[0]])
        
        act_a = a_log.act(input_a, np.array([target_cpu]))
        act_b = b_log.act(input_b, np.array([target_gpu]))
        act_c = c_log.act(input_c, np.array([0.5]))
        act_d = d_log.act(input_d, np.array([0.5]))
        
        comm_a = a_comm.act(input_a, np.array([target_cpu]))
        comm_b = b_comm.act(input_b, np.array([target_gpu]))
        comm_c = c_comm.act(input_c, np.array([0.5]))
        comm_d = d_comm.act(input_d, np.array([0.5]))
        
        act_a[0] = np.clip(act_a[0] + np.random.normal(0.0, 0.02), 0.0, 1.0)
        act_b[0] = np.clip(act_b[0] + np.random.normal(0.0, 0.02), 0.0, 1.0)
        act_c[0] = np.clip(act_c[0] + np.random.normal(0.0, 0.02), 0.0, 1.0)
        
        body.update(act_a[0], act_b[0], act_c[0])
        
        sym_a = np.clip(comm_a[0], 0.0, 1.0)
        sym_b = np.clip(comm_b[0], 0.0, 1.0)
        sym_c = np.clip(comm_c[0], 0.0, 1.0)
        sym_d = np.clip(comm_d[0], 0.0, 1.0)
        
        next_input_a = np.array([body.cpu, sym_b])
        next_input_b = np.array([body.gpu, sym_c])
        next_input_c = np.array([body.temp, sym_d])
        next_input_d = np.array([energy.energy, sym_a])
        
        a_log.learn(input_a, act_a, next_input_a)
        b_log.learn(input_b, act_b, next_input_b)
        c_log.learn(input_c, act_c, next_input_c)
        d_log.learn(input_d, act_d, next_input_d)
        
        a_comm.learn(input_a, comm_a, next_input_a)
        b_comm.learn(input_b, comm_b, next_input_b)
        c_comm.learn(input_c, comm_c, next_input_c)
        d_comm.learn(input_d, comm_d, next_input_d)
        
        last_msgs = [sym_a, sym_b, sym_c, sym_d]
        
        if energy.cpu_min <= body.cpu <= energy.cpu_max and energy.gpu_min <= body.gpu <= energy.gpu_max:
            energy.energy = min(1.0, energy.energy + 0.003)
        else:
            energy.energy -= 0.003
        if energy.energy <= 0:
            break
    
    fitness = (step + 1) + np.std(last_msgs) * 50
    return fitness, step + 1, energy.energy

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
            fit, life, energy_left = evaluate_specialization(brains)
            results.append((fit, life, brains, energy_left))
        results.sort(key=lambda x: x[0], reverse=True)
        
        best = results[0]
        if best_overall is None or best[0] > best_overall[0]:
            best_overall = best
            
        if gen % 5 == 0 or gen == generations - 1:
            print(f"  Поколение {gen}: Fitness={best[0]:.1f}, Жизнь={best[1]}, Энергия={best[3]:.3f}")
        
        new_pop = []
        for i in range(len(population)):
            parent = results[i % 3][2]
            child = tuple(copy.deepcopy(b) for b in parent)
            for b in child:
                b.W_s += np.random.normal(0, 0.05, b.W_s.shape)
                b.W_a += np.random.normal(0, 0.05, b.W_a.shape)
            new_pop.append(child)
        population = new_pop
    return population, best_overall

def main():
    print("Финальный эксперимент: 50 поколений с жесткой специализацией...\n")
    pop = create_population(10)
    pop, best = evolve(pop, 50)
    
    print(f"\n=== ЛУЧШИЙ АГЕНТ (50 ПОКОЛЕНИЙ) ===")
    print(f"  Fitness: {best[0]:.1f}")
    print(f"  Жизнь: {best[1]}")
    print(f"  Энергия: {best[3]:.3f}")

if __name__ == "__main__":
    main()
