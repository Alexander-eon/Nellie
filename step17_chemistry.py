import numpy as np
import copy
import pickle
import os
from predictive_brain_v2 import PredictiveBrainV2

GENE_FILE = "digital_gene_chemistry.pkl"

# === ХИМИЧЕСКИЕ ЭЛЕМЕНТЫ ===
ELEMENTS = {
    'H': np.array([1.0, 0.0, 1.0]),
    'O': np.array([16.0, 0.0, 2.0]),
    'C': np.array([12.0, 0.0, 4.0]),
    'N': np.array([14.0, 0.0, 3.0]),
    'Na': np.array([23.0, 1.0, 1.0]),
    'Cl': np.array([35.5, -1.0, 1.0]),
    'Fe': np.array([56.0, 2.0, 2.0]),
    'Ca': np.array([40.0, 2.0, 2.0]),
    'S': np.array([32.0, 0.0, 2.0]),
}

MOLECULES = {
    'H2O': ['H', 'H', 'O'],
    'CO2': ['C', 'O', 'O'],
    'NaCl': ['Na', 'Cl'],
    'CH4': ['C', 'H', 'H', 'H', 'H'],
    'NH3': ['N', 'H', 'H', 'H'],
    'Fe2O3': ['Fe', 'Fe', 'O', 'O', 'O'],
    'CaCO3': ['Ca', 'C', 'O', 'O', 'O'],
    'H2SO4': ['H', 'H', 'S', 'O', 'O', 'O', 'O'],
}

def molecule_to_vector(molecule_name):
    elements = MOLECULES[molecule_name]
    vector = np.zeros(3)
    for elem in elements:
        vector += ELEMENTS[elem]
    return vector

class ChemistryBrain:
    def __init__(self, n_sensors=12, n_actions=2):
        self.pred_brain = PredictiveBrainV2(n_sensors=n_sensors, n_actions=n_actions, lr=0.1, lr_action=0.1)
        
    def act(self, sensors, target):
        return self.pred_brain.act(sensors, target)
        
    def learn(self, sensors, action, next_sensors):
        self.pred_brain.learn(sensors, action, next_sensors)

def evaluate_pair(brains, steps=100):
    brain_a, brain_b = brains
    total_error = 0.0
    correct_transfers = 0
    energy = 1.0
    
    for step in range(steps):
        mol_name = np.random.choice(list(MOLECULES.keys()))
        mol_vector = molecule_to_vector(mol_name)
        mol_normalized = mol_vector / 100.0
        
        sensors_a = np.array(list(mol_normalized) + [0.0] * 7 + [0.0, 0.5])
        sensors_b = np.array(list(mol_normalized + np.random.normal(0, 0.01, 3)) + [0.0] * 7 + [0.0, 0.5])
        
        # Цель: предсказать следующее состояние (то же самое, что и текущее, т.к. молекула статична)
        target_a = np.array(list(mol_normalized) + [0.0] * 7 + [0.0, 0.5])
        target_b = target_a.copy()
        
        action_a = brain_a.act(sensors_a, target_a)
        action_b = brain_b.act(sensors_b, target_b)
        
        compressed_a = action_a[:2]
        
        mol_mean = mol_normalized.mean()
        compression_error = abs(compressed_a[0] - mol_mean)
        total_error += compression_error
        
        if abs(action_b[0] - action_a[0]) < 0.2:
            correct_transfers += 1
            energy = min(1.0, energy + 0.02)
        else:
            energy -= 0.01
        
        next_sensors_a = np.array(list(mol_normalized) + [0.0] * 7 + [0.0, 0.5])
        next_sensors_b = next_sensors_a.copy()
        
        brain_a.learn(sensors_a, action_a, next_sensors_a)
        brain_b.learn(sensors_b, action_b, next_sensors_b)
        
        if energy <= 0:
            break
    
    fitness = correct_transfers * 10 - total_error + energy * 100
    return fitness, steps, correct_transfers, total_error, energy

def save_gene(brain):
    with open(GENE_FILE, 'wb') as f:
        pickle.dump({'W_s': brain.pred_brain.W_s, 'W_a': brain.pred_brain.W_a}, f)

def load_gene():
    if os.path.exists(GENE_FILE):
        with open(GENE_FILE, 'rb') as f:
            data = pickle.load(f)
        brain = ChemistryBrain(n_sensors=12, n_actions=2)
        # Проверяем размерности перед загрузкой
        if data['W_s'].shape == brain.pred_brain.W_s.shape and data['W_a'].shape == brain.pred_brain.W_a.shape:
            brain.pred_brain.W_s = data['W_s']
            brain.pred_brain.W_a = data['W_a']
            return brain
        else:
            print("⚠️ Размерности гена не совпадают. Создаем нового агента.")
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
            brain = ChemistryBrain(n_sensors=12, n_actions=2)
        population.append(brain)
    
    for i in range(7):
        brain = ChemistryBrain(n_sensors=12, n_actions=2)
        brain.pred_brain.W_a += np.random.normal(0, 0.2, brain.pred_brain.W_a.shape)
        population.append(brain)
    
    best_overall = None
    best_fitness = -999
    
    for gen in range(20):
        results = []
        pairs = [(population[i], population[(i+1) % len(population)]) for i in range(len(population))]
        
        for brain_a, brain_b in pairs:
            fitness, life, correct, error, energy = evaluate_pair((brain_a, brain_b), steps=50)
            results.append((fitness, life, brain_a, brain_b, correct, error, energy))
        
        results.sort(key=lambda x: x[0], reverse=True)
        best = results[0]
        
        if best[0] > best_fitness:
            best_fitness = best[0]
            best_overall = copy.deepcopy(best[2])
            save_gene(best_overall)
            print(f"  Поколение {gen}: Fitness={best[0]:.1f}, Правильно={best[4]}, Ошибка={best[5]:.2f}, Энергия={best[6]:.2f} <-- ЛУЧШИЙ")
        else:
            print(f"  Поколение {gen}: Fitness={best[0]:.1f}, Правильно={best[4]}, Ошибка={best[5]:.2f}, Энергия={best[6]:.2f}")
        
        new_population = []
        top3 = [results[0][2], results[1][2], results[2][2]]
        for i in range(10):
            parent1 = top3[np.random.randint(0, 3)]
            parent2 = top3[np.random.randint(0, 3)]
            child = ChemistryBrain(n_sensors=12, n_actions=2)
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
    print("=== ХИМИЯ (20 ПОКОЛЕНИЙ) ===\n")
    best_brain = evolve_population()
    
    print("\n=== ФИНАЛЬНЫЙ ТЕСТ ===")
    partner = copy.deepcopy(best_brain)
    fitness, life, correct, error, energy = evaluate_pair((best_brain, partner), steps=100)
    print(f"Прожито тактов: {life}")
    print(f"Правильно передано: {correct}/100")
    print(f"Ошибка: {error:.2f}")
    print(f"Энергия: {energy:.2f}")
    print(f"Fitness: {fitness:.1f}")

if __name__ == "__main__":
    main()
