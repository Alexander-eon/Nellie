import numpy as np
import copy
import pickle
import os
from predictive_brain_v2 import PredictiveBrainV2

GENE_FILE = "digital_gene_concepts.pkl"

# === СЛОВАРЬ ПОНЯТИЙ (симуляция LLM) ===
# Каждое слово имеет эмбеддинг (10 чисел) — это "связи" из LLM
# И правильное действие (0.0 - 1.0) — как реагировать
CONCEPTS = {
    'собака': {'embedding': np.array([0.8, 0.2, 0.6, 0.1, 0.9, 0.3, 0.7, 0.4, 0.2, 0.5]), 'action': 0.8},
    'еда': {'embedding': np.array([0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.5, 0.1]), 'action': 0.2},
    'опасность': {'embedding': np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.9, 0.1, 0.6, 0.4]), 'action': 0.9},
    'друг': {'embedding': np.array([0.2, 0.8, 0.1, 0.9, 0.3, 0.7, 0.1, 0.8, 0.4, 0.6]), 'action': 0.1},
    'боль': {'embedding': np.array([0.7, 0.3, 0.9, 0.1, 0.8, 0.2, 0.6, 0.4, 0.9, 0.3]), 'action': 0.9},
    'голод': {'embedding': np.array([0.3, 0.7, 0.2, 0.8, 0.1, 0.9, 0.3, 0.7, 0.2, 0.8]), 'action': 0.2},
    'тепло': {'embedding': np.array([0.4, 0.6, 0.5, 0.5, 0.4, 0.6, 0.5, 0.5, 0.4, 0.6]), 'action': 0.5},
    'холод': {'embedding': np.array([0.6, 0.4, 0.5, 0.5, 0.6, 0.4, 0.5, 0.5, 0.6, 0.4]), 'action': 0.5},
    'движение': {'embedding': np.array([0.5, 0.5, 0.7, 0.3, 0.5, 0.5, 0.7, 0.3, 0.5, 0.5]), 'action': 0.7},
    'покой': {'embedding': np.array([0.5, 0.5, 0.3, 0.7, 0.5, 0.5, 0.3, 0.7, 0.5, 0.5]), 'action': 0.3},
}

class ConceptBrain:
    """Мозг, который использует эмбеддинги понятий для выживания"""
    def __init__(self, n_sensors=12, n_actions=1):
        # 12 сенсоров: 10 эмбеддинг + боль + голод
        # 1 действие: реакция (0.0 - 1.0)
        self.pred_brain = PredictiveBrainV2(n_sensors=n_sensors, n_actions=n_actions, lr=0.1, lr_action=0.1)
        
    def act(self, sensors, target):
        return self.pred_brain.act(sensors, target)
        
    def learn(self, sensors, action, next_sensors):
        self.pred_brain.learn(sensors, action, next_sensors)

def evaluate_agent(brain, steps=100):
    total_error = 0.0
    correct_actions = 0
    energy = 1.0
    
    for step in range(steps):
        # Выбираем случайное понятие
        concept_name = np.random.choice(list(CONCEPTS.keys()))
        concept = CONCEPTS[concept_name]
        
        # Симулируем "боль" и "голод" (случайные)
        pain = np.random.rand()
        hunger = np.random.rand()
        
        # Сенсоры: эмбеддинг понятия + боль + голод
        sensors = np.array(list(concept['embedding']) + [pain, hunger])
        
        # Правильное действие: зависит от понятия и контекста
        # Если понятие "опасность" и боль высокая -> действуй (0.9)
        # Если понятие "еда" и голод высокий -> действуй (0.2)
        # Если понятие "друг" -> действуй (0.1)
        if concept_name in ['опасность', 'боль', 'собака']:
            target_action = concept['action'] * (0.5 + pain * 0.5)  # Чем больше боль, тем сильнее реакция
        elif concept_name in ['еда', 'голод']:
            target_action = concept['action'] * (0.5 + hunger * 0.5)  # Чем больше голод, тем сильнее реакция
        else:
            target_action = concept['action']
        
        # Агент действует
        target_sensors = sensors.copy()  # Цель для предсказания (стабильность)
        action = brain.act(sensors, target_sensors)
        action_value = np.clip(action[0], 0.0, 1.0)
        
        # Ошибка
        error = abs(action_value - target_action)
        total_error += error
        
        if error < 0.2:
            correct_actions += 1
            energy = min(1.0, energy + 0.02)
        else:
            energy -= 0.01
        
        # Обучение
        next_sensors = sensors.copy()
        brain.learn(sensors, action, next_sensors)
        
        if energy <= 0:
            break
    
    fitness = correct_actions * 10 - total_error + energy * 100
    return fitness, steps, correct_actions, total_error, energy

def save_gene(brain):
    with open(GENE_FILE, 'wb') as f:
        pickle.dump({'W_s': brain.pred_brain.W_s, 'W_a': brain.pred_brain.W_a}, f)

def load_gene():
    if os.path.exists(GENE_FILE):
        with open(GENE_FILE, 'rb') as f:
            data = pickle.load(f)
        brain = ConceptBrain(n_sensors=12, n_actions=1)
        if data['W_s'].shape == brain.pred_brain.W_s.shape:
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
            brain = ConceptBrain(n_sensors=12, n_actions=1)
        population.append(brain)
    
    for i in range(7):
        brain = ConceptBrain(n_sensors=12, n_actions=1)
        brain.pred_brain.W_a += np.random.normal(0, 0.2, brain.pred_brain.W_a.shape)
        population.append(brain)
    
    best_overall = None
    best_fitness = -999
    
    for gen in range(30):
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
            child = ConceptBrain(n_sensors=12, n_actions=1)
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
    print("=== ПОНЯТИЙНЫЙ АППАРАТ (LLM КАК СПРАВОЧНИК) - 30 ПОКОЛЕНИЙ ===\n")
    print("10 понятий, 10 эмбеддингов, 2 сенсора (боль, голод)\n")
    best_brain = evolve_population()
    
    print("\n=== ФИНАЛЬНЫЙ ТЕСТ ===")
    fitness, life, correct, error, energy = evaluate_agent(best_brain, steps=100)
    print(f"Прожито тактов: {life}")
    print(f"Правильно действий: {correct}/100")
    print(f"Ошибка: {error:.2f}")
    print(f"Энергия: {energy:.2f}")
    print(f"Fitness: {fitness:.1f}")

if __name__ == "__main__":
    main()
