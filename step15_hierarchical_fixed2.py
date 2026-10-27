import numpy as np
import copy
import pickle
import os
from predictive_brain import PredictiveBrain

GENE_FILE = "digital_gene_hierarchical2.pkl"

def generate_frames():
    frames = []
    frame = np.zeros((10, 10)); frame[5, :] = 1.0; frames.append(frame.flatten())
    frame = np.zeros((10, 10)); frame[:, 5] = 1.0; frames.append(frame.flatten())
    frame = np.zeros((10, 10)); frame[2:8, 2:8] = 1.0; frames.append(frame.flatten())
    frame = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            if (i-5)**2 + (j-5)**2 < 9: frame[i, j] = 1.0
    frames.append(frame.flatten())
    frame = np.zeros((10, 10))
    for i in range(10): frame[i, i] = 1.0
    frames.append(frame.flatten())
    frame = np.zeros((10, 10)); frame[5, :] = 1.0; frame[:, 5] = 1.0; frames.append(frame.flatten())
    frame = np.zeros((10, 10))
    for i in range(5): frame[5+i, 5-i:5+i+1] = 1.0
    frames.append(frame.flatten())
    frame = np.zeros((10, 10)); frame[1, 1:9] = 1.0; frame[9, 1:9] = 1.0; frame[1:9, 1] = 1.0; frame[1:9, 9] = 1.0; frames.append(frame.flatten())
    frame = np.zeros((10, 10)); frame[5, 5] = 1.0; frames.append(frame.flatten())
    frame = np.zeros((10, 10)); frame[3, 3] = 1.0; frame[7, 7] = 1.0; frames.append(frame.flatten())
    return frames

FRAMES = generate_frames()

class CompressionBrain:
    def __init__(self):
        self.neurons = [PredictiveBrain(n_sensors=12, n_actions=1, lr=0.1, lr_action=0.1) for _ in range(10)]
        
    def act(self, sensors):
        actions = []
        for neuron in self.neurons:
            action = neuron.act(sensors, np.array([0.5]))
            actions.append(np.clip(action[0], 0.0, 1.0))
        return np.array(actions)
        
    def learn(self, sensors, actions, next_sensors):
        for i, neuron in enumerate(self.neurons):
            neuron.learn(sensors, np.array([actions[i]]), next_sensors)

def evaluate_pair(brains, steps=100):
    brain_a, brain_b = brains
    total_error = 0.0
    total_energy_cost = 0.0
    consensus_hits = 0
    active_counts = []
    information_transfer = 0  # Сколько информации передано
    
    for step in range(steps):
        frame_idx = np.random.randint(0, len(FRAMES))
        frame = FRAMES[frame_idx]
        
        compressed = frame.reshape(10, 10).mean(axis=1)
        sensors = np.array(list(compressed) + [0.0, 0.5])
        
        action_a = brain_a.act(sensors)
        action_b = brain_b.act(sensors)
        
        active_a = np.sum(action_a > 0.1)
        active_b = np.sum(action_b > 0.1)
        energy_cost = (active_a + active_b) * 0.01
        total_energy_cost += energy_cost
        active_counts.append(active_a)
        
        # ШТРАФ ЗА ПУСТОТУ: если агент не передал ничего (0 чисел) — он не передал информацию
        if active_a == 0:
            information_transfer -= 0.5  # Штраф за пустоту
        
        # Консенсус
        distance = np.linalg.norm(action_a - action_b)
        total_error += distance
        if distance < 0.2:
            consensus_hits += 1
            # Бонус за консенсус только если есть информация
            if active_a > 0 and active_b > 0:
                information_transfer += 1
        
        brain_a.learn(sensors, action_a, sensors)
        brain_b.learn(sensors, action_b, sensors)
    
    # FITNESS: штраф за энергию + бонус за информацию - ошибка
    fitness = information_transfer * 10 - total_error - total_energy_cost * 5
    avg_active = np.mean(active_counts) if active_counts else 0
    return fitness, steps, total_error, consensus_hits, total_energy_cost, avg_active, information_transfer

def save_gene(brain):
    weights = []
    for neuron in brain.neurons:
        weights.append({'W_s': neuron.W_s, 'W_a': neuron.W_a})
    with open(GENE_FILE, 'wb') as f:
        pickle.dump(weights, f)

def load_gene():
    if os.path.exists(GENE_FILE):
        with open(GENE_FILE, 'rb') as f:
            weights = pickle.load(f)
        brain = CompressionBrain()
        for i, w in enumerate(weights):
            if i < len(brain.neurons):
                brain.neurons[i].W_s = w['W_s']
                brain.neurons[i].W_a = w['W_a']
        return brain
    return None

def evolve_population():
    population = []
    innate_brain = load_gene()
    
    for i in range(3):
        if innate_brain:
            brain = copy.deepcopy(innate_brain)
            for neuron in brain.neurons:
                neuron.W_s += np.random.normal(0, 0.1, neuron.W_s.shape)
                neuron.W_a += np.random.normal(0, 0.1, neuron.W_a.shape)
        else:
            brain = CompressionBrain()
        population.append(brain)
    
    for i in range(7):
        brain = CompressionBrain()
        for neuron in brain.neurons:
            neuron.W_a += np.random.normal(0, 0.2, neuron.W_a.shape)
        population.append(brain)
    
    best_overall = None
    best_fitness = -999
    
    for gen in range(20):
        results = []
        pairs = [(population[i], population[(i+1) % len(population)]) for i in range(len(population))]
        
        for brain_a, brain_b in pairs:
            fitness, life, error, consensus, energy_cost, avg_active, info = evaluate_pair((brain_a, brain_b), steps=50)
            results.append((fitness, life, brain_a, brain_b, error, consensus, energy_cost, avg_active, info))
        
        results.sort(key=lambda x: x[0], reverse=True)
        best = results[0]
        
        if best[0] > best_fitness:
            best_fitness = best[0]
            best_overall = copy.deepcopy(best[2])
            save_gene(best_overall)
            print(f"  Поколение {gen}: Fitness={best[0]:.1f}, Ошибка={best[4]:.2f}, Инфо={best[8]}, Энергия={best[6]:.2f}, Чисел={best[7]:.1f} <-- ЛУЧШИЙ")
        else:
            print(f"  Поколение {gen}: Fitness={best[0]:.1f}, Ошибка={best[4]:.2f}, Инфо={best[8]}, Энергия={best[6]:.2f}, Чисел={best[7]:.1f}")
        
        new_population = []
        top3 = [results[0][2], results[1][2], results[2][2]]
        for i in range(10):
            parent1 = top3[np.random.randint(0, 3)]
            parent2 = top3[np.random.randint(0, 3)]
            child = CompressionBrain()
            for j in range(10):
                mask = np.random.rand(*parent1.neurons[j].W_s.shape) < 0.5
                child.neurons[j].W_s = np.where(mask, parent1.neurons[j].W_s, parent2.neurons[j].W_s)
                mask_a = np.random.rand(*parent1.neurons[j].W_a.shape) < 0.5
                child.neurons[j].W_a = np.where(mask_a, parent1.neurons[j].W_a, parent2.neurons[j].W_a)
                child.neurons[j].W_s += np.random.normal(0, 0.1, child.neurons[j].W_s.shape)
                child.neurons[j].W_a += np.random.normal(0, 0.1, child.neurons[j].W_a.shape)
            new_population.append(child)
        population = new_population
    return best_overall

def main():
    print("=== ИЕРАРХИЧЕСКОЕ СЖАТИЕ С ШТРАФОМ ЗА ПУСТОТУ ===\n")
    best_brain = evolve_population()
    
    print("\n=== ФИНАЛЬНЫЙ ТЕСТ ===")
    partner = copy.deepcopy(best_brain)
    fitness, life, error, consensus, energy_cost, avg_active, info = evaluate_pair((best_brain, partner), steps=100)
    print(f"Прожито тактов: {life}")
    print(f"Ошибка: {error:.2f}")
    print(f"Консенсус: {consensus}")
    print(f"Энергозатраты: {energy_cost:.2f}")
    print(f"Среднее число активных чисел: {avg_active:.1f}")
    print(f"Передано информации: {info}")
    print(f"Fitness: {fitness:.1f}")

if __name__ == "__main__":
    main()
