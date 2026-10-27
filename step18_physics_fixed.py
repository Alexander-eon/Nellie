import numpy as np
import copy
import pickle
import os
from predictive_brain_v2 import PredictiveBrainV2

GENE_FILE = "digital_gene_physics.pkl"

class PhysicsWorld:
    def __init__(self):
        self.x = 0.5
        self.y = 0.8
        self.vx = 0.1
        self.vy = 0.0
        self.gravity = -0.1
        self.dt = 1.0
        
    def step(self):
        self.vy += self.gravity * self.dt
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt
        if self.y < 0.0:
            self.y = -self.y
            self.vy = -self.vy * 0.8
        if self.x < 0.0 or self.x > 1.0:
            self.vx = -self.vx
            self.x = np.clip(self.x, 0.0, 1.0)
            
    def get_sensors(self):
        return np.array([self.x, self.y, self.vx, self.vy])

class PhysicsBrain:
    def __init__(self, n_sensors=6, n_actions=1):
        # 6 сенсоров: x, y, vx, vy, боль, шум
        # 1 действие: предсказание "ошибки" (насколько агент уверен)
        self.pred_brain = PredictiveBrainV2(n_sensors=n_sensors, n_actions=n_actions, lr=0.1, lr_action=0.1)
        
    def act(self, sensors, target):
        return self.pred_brain.act(sensors, target)
        
    def learn(self, sensors, action, next_sensors):
        self.pred_brain.learn(sensors, action, next_sensors)

def evaluate_agent(brain, steps=100):
    world = PhysicsWorld()
    total_error = 0.0
    correct_predictions = 0
    energy = 1.0
    
    for step in range(steps):
        sensors = np.array(list(world.get_sensors()) + [0.0, 0.5])
        
        # Цель: предсказать следующее состояние (6 сенсоров)
        # Мы даем агенту "идеальное" следующее состояние как цель
        old_state = world.get_sensors().copy()
        world.step()
        next_state = world.get_sensors()
        target = np.array(list(next_state) + [0.0, 0.5])
        
        action = brain.act(sensors, target)
        
        # Оценка: насколько хорошо агент предсказал следующее состояние
        pred = brain.pred_brain.predict(sensors, action)
        error = np.mean(np.abs(pred - target))
        total_error += error
        
        if error < 0.2:
            correct_predictions += 1
            energy = min(1.0, energy + 0.02)
        else:
            energy -= 0.01
        
        next_sensors = np.array(list(world.get_sensors()) + [0.0, 0.5])
        brain.learn(sensors, action, next_sensors)
        
        if energy <= 0:
            break
    
    fitness = correct_predictions * 10 - total_error + energy * 100
    return fitness, steps, correct_predictions, total_error, energy

def save_gene(brain):
    with open(GENE_FILE, 'wb') as f:
        pickle.dump({'W_s': brain.pred_brain.W_s, 'W_a': brain.pred_brain.W_a}, f)

def load_gene():
    if os.path.exists(GENE_FILE):
        with open(GENE_FILE, 'rb') as f:
            data = pickle.load(f)
        brain = PhysicsBrain(n_sensors=6, n_actions=1)
        if data['W_s'].shape == brain.pred_brain.W_s.shape and data['W_a'].shape == brain.pred_brain.W_a.shape:
            brain.pred_brain.W_s = data['W_s']
            brain.pred_brain.W_a = data['W_a']
            return brain
    return None

def evolve_population():
    population = []
    innate_brain = load_gene()
    
    for i in range(3):
        if innate_brain:
            brain = copy.deepcopy(innate_brain)
            brain.pred_brain.W_s += np.random.normal(0, 0.1, brain.pred_brain.W_s.shape)
            brain.pred_brain.W_a += np.random.normal(0, 0.1, brain.pred_brain.W_a.shape)
        else:
            brain = PhysicsBrain(n_sensors=6, n_actions=1)
        population.append(brain)
    
    for i in range(7):
        brain = PhysicsBrain(n_sensors=6, n_actions=1)
        brain.pred_brain.W_a += np.random.normal(0, 0.2, brain.pred_brain.W_a.shape)
        population.append(brain)
    
    best_overall = None
    best_fitness = -999
    
    for gen in range(20):
        results = []
        for brain in population:
            fitness, life, correct, error, energy = evaluate_agent(brain, steps=100)
            results.append((fitness, life, brain, correct, error, energy))
        
        results.sort(key=lambda x: x[0], reverse=True)
        best = results[0]
        
        if best[0] > best_fitness:
            best_fitness = best[0]
            best_overall = copy.deepcopy(best[2])
            save_gene(best_overall)
            print(f"  Поколение {gen}: Fitness={best[0]:.1f}, Правильно={best[3]}, Ошибка={best[4]:.2f}, Энергия={best[5]:.2f} <-- ЛУЧШИЙ")
        else:
            print(f"  Поколение {gen}: Fitness={best[0]:.1f}, Правильно={best[3]}, Ошибка={best[4]:.2f}, Энергия={best[5]:.2f}")
        
        new_population = []
        top3 = [results[0][2], results[1][2], results[2][2]]
        for i in range(10):
            parent1 = top3[np.random.randint(0, 3)]
            parent2 = top3[np.random.randint(0, 3)]
            child = PhysicsBrain(n_sensors=6, n_actions=1)
            mask = np.random.rand(*parent1.pred_brain.W_s.shape) < 0.5
            child.pred_brain.W_s = np.where(mask, parent1.pred_brain.W_s, parent2.pred_brain.W_s)
            mask_a = np.random.rand(*parent1.pred_brain.W_a.shape) < 0.5
            child.pred_brain.W_a = np.where(mask_a, parent1.pred_brain.W_a, parent2.pred_brain.W_a)
            child.pred_brain.W_s += np.random.normal(0, 0.1, child.pred_brain.W_s.shape)
            child.pred_brain.W_a += np.random.normal(0, 0.1, child.pred_brain.W_a.shape)
            new_population.append(child)
        population = new_population
    return best_overall

def main():
    print("=== ФИЗИКА (20 ПОКОЛЕНИЙ) ===\n")
    best_brain = evolve_population()
    
    print("\n=== ФИНАЛЬНЫЙ ТЕСТ ===")
    fitness, life, correct, error, energy = evaluate_agent(best_brain, steps=100)
    print(f"Прожито тактов: {life}")
    print(f"Правильно предсказано: {correct}/100")
    print(f"Ошибка: {error:.2f}")
    print(f"Энергия: {energy:.2f}")
    print(f"Fitness: {fitness:.1f}")

if __name__ == "__main__":
    main()
