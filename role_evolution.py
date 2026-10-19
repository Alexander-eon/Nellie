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

def evaluate_roles(brains, steps=300):
    # У нас 4 агента, каждый видит ВСЕ 4 сенсора и управляет ВСЕМИ 4 действиями
    # Но у каждого есть только одно "действие" (свой выход)
    a_log, a_comm, b_log, b_comm, c_log, c_comm, d_log, d_comm = brains
    body = VirtualBody()
    energy = Energy(initial=1.0, drain_rate=0.003, bounty_rate=0.003, 
                    cpu_min=0.40, cpu_max=0.60, gpu_min=0.40, gpu_max=0.60)
    
    last_msgs = [0.5, 0.5, 0.5, 0.5] # Сообщения от всех агентов
    
    for step in range(steps):
        t = step / 50.0
        target_cpu = 0.5 + 0.15 * np.sin(t)
        target_gpu = 0.5 + 0.15 * np.cos(t)
        
        # Каждые 50 шагов — кризис
        if step % 50 == 0 and step > 0:
            energy.energy -= 0.2
        
        # Все агенты видят ВСЕ сенсоры
        sensors = np.array([body.cpu, body.gpu, body.temp, energy.energy])
        
        # Каждый агент получает все сенсоры + одно сообщение от партнера (по кругу)
        input_a = np.concatenate([sensors, [last_msgs[1]]])
        input_b = np.concatenate([sensors, [last_msgs[2]]])
        input_c = np.concatenate([sensors, [last_msgs[3]]])
        input_d = np.concatenate([sensors, [last_msgs[0]]])
        
        # Каждый агент управляет ВСЕМИ аспектами (но у него только 1 выход)
        # Агент А хочет держать CPU, B — GPU, C — Temp, D — Energy
        act_a = a_log.act(input_a, np.array([target_cpu]))
        act_b = b_log.act(input_b, np.array([target_gpu]))
        act_c = c_log.act(input_c, np.array([0.5]))
        act_d = d_log.act(input_d, np.array([0.5]))
        
        # Коммуникация: каждый передает сигнал о своем состоянии
        comm_a = a_comm.act(input_a, np.array([target_cpu]))
        comm_b = b_comm.act(input_b, np.array([target_gpu]))
        comm_c = c_comm.act(input_c, np.array([0.5]))
        comm_d = d_comm.act(input_d, np.array([0.5]))
        
        # Собираем действия (каждый агент влияет на свою часть тела)
        act_a[0] = np.clip(act_a[0] + np.random.normal(0.0, 0.02), 0.0, 1.0)
        act_b[0] = np.clip(act_b[0] + np.random.normal(0.0, 0.02), 0.0, 1.0)
        act_c[0] = np.clip(act_c[0] + np.random.normal(0.0, 0.02), 0.0, 1.0)
        act_d[0] = np.clip(act_d[0] + np.random.normal(0.0, 0.02), 0.0, 1.0)
        
        body.update(act_a[0], act_b[0], act_c[0])
        
        # Сообщения
        sym_a = np.clip(comm_a[0], 0.0, 1.0)
        sym_b = np.clip(comm_b[0], 0.0, 1.0)
        sym_c = np.clip(comm_c[0], 0.0, 1.0)
        sym_d = np.clip(comm_d[0], 0.0, 1.0)
        
        # Обучение
        next_sensors = np.array([body.cpu, body.gpu, body.temp, energy.energy])
        next_input_a = np.concatenate([next_sensors, [sym_b]])
        next_input_b = np.concatenate([next_sensors, [sym_c]])
        next_input_c = np.concatenate([next_sensors, [sym_d]])
        next_input_d = np.concatenate([next_sensors, [sym_a]])
        
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
    
    # Fitness: выживание + бонус за то, что все агенты передают разные сигналы (разделение ролей)
    # Если все передают одно и то же — это плохо (нет разделения)
    if len(last_msgs) == 4:
        diversity = np.std(last_msgs) # Разброс сообщений
    else:
        diversity = 0
    
    fitness = (step + 1) + diversity * 100 # Бонус за разнообразие ролей
    return fitness, step + 1, energy.energy

def create_population(size):
    pop = []
    for _ in range(size):
        # Каждый агент видит 5 входов (4 сенсора + 1 сообщение)
        brains = (
            PredictiveBrain(5, 1), PredictiveBrain(5, 1),
            PredictiveBrain(5, 1), PredictiveBrain(5, 1),
            PredictiveBrain(5, 1), PredictiveBrain(5, 1),
            PredictiveBrain(5, 1), PredictiveBrain(5, 1)
        )
        pop.append(brains)
    return pop

def evolve(population, generations):
    best_overall = None
    for gen in range(generations):
        results = []
        for brains in population:
            fit, life, energy_left = evaluate_roles(brains)
            results.append((fit, life, brains, energy_left))
        results.sort(key=lambda x: x[0], reverse=True)
        
        best = results[0]
        if best_overall is None or best[0] > best_overall[0]:
            best_overall = best
            
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
    print("Эволюция РОЛЕЙ (4 агента, все видят все сенсоры)...\n")
    print("Цель: посмотреть, как агенты сами распределят роли.\n")
    pop = create_population(10)
    pop, best = evolve(pop, 20)
    
    print(f"\n=== ЛУЧШИЙ АГЕНТ (20 ПОКОЛЕНИЙ) ===")
    print(f"  Fitness: {best[0]:.1f}")
    print(f"  Жизнь: {best[1]}")
    print(f"  Энергия: {best[3]:.3f}")

if __name__ == "__main__":
    main()
