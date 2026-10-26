import numpy as np
import copy
import pickle
import os
from predictive_brain import PredictiveBrain

GENE_FILE = "digital_gene_echo.pkl"

class VirtualBody:
    def __init__(self, initial_temp=0.1):
        self.temp = initial_temp
        self.pain = 0.0
        self.fear = 0.0
        self.energy = 1.0
        self.hunger = 0.0
        self.sleepiness = 0.0
        
    def update(self, action):
        heat = (action - 0.5) * 0.3
        self.temp = np.clip(self.temp + heat, 0.0, 1.0)
        if self.temp > 0.7:
            self.pain = (self.temp - 0.7) / 0.3
        elif self.temp < 0.3:
            self.pain = (0.3 - self.temp) / 0.3
        else:
            self.pain = 0.0
        
        self.hunger = 1.0 - self.energy
        self.sleepiness = min(1.0, self.sleepiness + 0.01)

class EchoBrain:
    def __init__(self, n_sensors=9, n_actions=1):
        # 9 сенсоров: temp, pain, fear, energy, hunger, sleepiness, 
        #             эхо (ответ мира), боль партнера, шум
        self.pred_brain = PredictiveBrain(n_sensors=n_sensors, n_actions=n_actions, lr=0.1, lr_action=0.1)
        self.trauma_memory = []
        self.seen_states = set()
        
    def act(self, sensors, target, pain_level, current_temp, hunger, sleepiness):
        fear = 0.0
        current_state = tuple(np.round([current_temp, sensors[1], sensors[2]], 1))
        for trauma in self.trauma_memory:
            distance = sum(abs(a - b) for a, b in zip(current_state, trauma['state']))
            if distance < 0.3:
                fear = max(fear, trauma['pain'])
        
        curiosity_bonus = 0.0
        if current_state not in self.seen_states:
            self.seen_states.add(current_state)
            curiosity_bonus = 0.05
            if len(self.seen_states) > 500:
                self.seen_states = set(list(self.seen_states)[-500:])
        
        if sleepiness > 0.7:
            target = np.array([0.5])
        elif hunger > 0.5:
            target = np.array([0.6 + hunger * 0.2])
        elif fear > 0.3:
            if current_temp < 0.5:
                target = np.array([0.6 + fear * 0.2])
            else:
                target = np.array([0.4 - fear * 0.2])
        else:
            target = np.array([0.5])
        
        action = self.pred_brain.act(sensors, target)
        return np.clip(action[0], 0.0, 1.0), fear, curiosity_bonus
        
    def learn(self, sensors, action, next_sensors, pain_level):
        self.pred_brain.learn(sensors, np.array([action]), next_sensors)
        if pain_level > 0.8:
            state = tuple(np.round(sensors[:3], 1))
            for trauma in self.trauma_memory:
                if trauma['state'] == state:
                    trauma['pain'] = max(trauma['pain'], pain_level)
                    return
            self.trauma_memory.append({'state': state, 'pain': pain_level})
            if len(self.trauma_memory) > 20:
                self.trauma_memory.pop(0)
    
    def consolidate_memory(self):
        self.trauma_memory = [t for t in self.trauma_memory if t['pain'] > 0.3]
        if len(self.seen_states) > 200:
            self.seen_states = set(list(self.seen_states)[-200:])

def evaluate_agent(brain, steps=100):
    body = VirtualBody(initial_temp=0.1)
    total_pain = 0.0
    total_echo_perception = 0.0
    echo_history = []  # История сигналов (для эха)
    
    for step in range(steps):
        # Ночь каждые 20 шагов
        if step % 20 == 0 and step > 0:
            brain.consolidate_memory()
            body.energy = min(1.0, body.energy + 0.05)
            body.sleepiness = max(0.0, body.sleepiness - 0.5)
            continue
        
        # ЭХО: если есть история, возвращаем сигнал с задержкой 2 шага
        if len(echo_history) >= 2:
            echo_signal = echo_history[-2]  # Задержка 2 шага
            echo_signal += np.random.normal(0, 0.1)  # Шум
            echo_signal = np.clip(echo_signal, 0.0, 1.0)
        else:
            echo_signal = 0.5  # Нет эха (тишина)
        
        # Сенсоры: temp, pain, fear, energy, hunger, sleepiness, 
        #          эхо, боль партнера (имитация), шум
        sensors = np.array([
            body.temp, body.pain, body.fear, body.energy, 
            body.hunger, body.sleepiness, echo_signal, 0.5, 0.5
        ])
        
        target = np.array([0.5])
        action, fear, curiosity = brain.act(sensors, target, body.pain, body.temp, body.hunger, body.sleepiness)
        body.fear = fear
        
        if body.pain < 0.1:
            body.fear = max(0.0, body.fear - 0.1)
        
        body.update(action)
        echo_history.append(action)  # Запоминаем сигнал для эха
        
        total_pain += body.pain
        # Оценка восприятия эха: насколько агент использует эхо для предсказания
        if len(echo_history) > 2:
            correlation = np.corrcoef(echo_history[-2:], [action, echo_signal])[0, 1]
            if not np.isnan(correlation):
                total_echo_perception += abs(correlation)
        
        body.energy -= 0.005 + body.pain * 0.02 + body.hunger * 0.01
        
        if 0.4 <= body.temp <= 0.6:
            body.energy = min(1.0, body.energy + 0.01)
        
        next_sensors = np.array([
            body.temp, body.pain, body.fear, body.energy, 
            body.hunger, body.sleepiness, echo_signal, 0.5, 0.5
        ])
        brain.learn(sensors, action, next_sensors, body.pain)
        
        if body.energy <= 0:
            break
    
    fitness = (step + 1) - (total_pain / max(1, steps)) * 100 + (total_echo_perception / max(1, steps)) * 20
    return fitness, step + 1, body.energy, total_pain, total_echo_perception

def save_gene(brain):
    with open(GENE_FILE, 'wb') as f:
        pickle.dump({
            'W_s': brain.pred_brain.W_s,
            'W_a': brain.pred_brain.W_a,
            'trauma_memory': brain.trauma_memory
        }, f)

def load_gene():
    if os.path.exists(GENE_FILE):
        with open(GENE_FILE, 'rb') as f:
            data = pickle.load(f)
        brain = EchoBrain(n_sensors=9, n_actions=1)
        brain.pred_brain.W_s = data['W_s']
        brain.pred_brain.W_a = data['W_a']
        brain.trauma_memory = data.get('trauma_memory', [])
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
            brain = EchoBrain(n_sensors=9, n_actions=1)
        population.append(brain)
    
    for i in range(7):
        brain = EchoBrain(n_sensors=9, n_actions=1)
        brain.pred_brain.W_a += np.random.normal(0, 0.2, brain.pred_brain.W_a.shape)
        population.append(brain)
    
    best_overall = None
    best_fitness = -999
    
    for gen in range(20):
        results = []
        for brain in population:
            fitness, life, energy, pain, echo_perception = evaluate_agent(brain, steps=100)
            results.append((fitness, life, brain, energy, pain, echo_perception))
        
        results.sort(key=lambda x: x[0], reverse=True)
        best = results[0]
        
        if best[0] > best_fitness:
            best_fitness = best[0]
            best_overall = copy.deepcopy(best[2])
            save_gene(best_overall)
            print(f"  Поколение {gen}: Fitness={best[0]:.1f}, Боль={best[4]:.2f}, Эхо-восприятие={best[5]:.2f} <-- ЛУЧШИЙ")
        else:
            print(f"  Поколение {gen}: Fitness={best[0]:.1f}, Боль={best[4]:.2f}, Эхо-восприятие={best[5]:.2f}")
        
        # Кроссовер
        new_population = []
        top3 = [results[0][2], results[1][2], results[2][2]]
        for i in range(10):
            parent1 = top3[np.random.randint(0, 3)]
            parent2 = top3[np.random.randint(0, 3)]
            child = EchoBrain(n_sensors=9, n_actions=1)
            mask = np.random.rand(*parent1.pred_brain.W_s.shape) < 0.5
            child.pred_brain.W_s = np.where(mask, parent1.pred_brain.W_s, parent2.pred_brain.W_s)
            mask_a = np.random.rand(*parent1.pred_brain.W_a.shape) < 0.5
            child.pred_brain.W_a = np.where(mask_a, parent1.pred_brain.W_a, parent2.pred_brain.W_a)
            child.pred_brain.W_s += np.random.normal(0, 0.1, child.pred_brain.W_s.shape)
            child.pred_brain.W_a += np.random.normal(0, 0.1, child.pred_brain.W_a.shape)
            child.trauma_memory = copy.deepcopy(parent1.trauma_memory)
            new_population.append(child)
        population = new_population
    return best_overall

def main():
    print("=== ЭХО (20 ПОКОЛЕНИЙ) ===\n")
    best_brain = evolve_population()
    
    print("\n=== ФИНАЛЬНЫЙ ТЕСТ ===")
    fitness, life, energy, pain, echo_perception = evaluate_agent(best_brain, steps=200)
    print(f"Прожито тактов: {life}")
    print(f"Остаток энергии: {energy:.3f}")
    print(f"Суммарная боль: {pain:.2f}")
    print(f"Восприятие эха: {echo_perception:.2f}")
    print(f"Fitness: {fitness:.1f}")

if __name__ == "__main__":
    main()
