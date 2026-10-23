import numpy as np
import copy
import pickle
import os
from predictive_brain import PredictiveBrain

GENE_FILE = "digital_gene_evolution.pkl"

class VirtualBody:
    def __init__(self, initial_temp=0.1, lifespan=500):
        self.temp = initial_temp
        self.pain = 0.0
        self.fear = 0.0
        self.energy = 1.0
        self.hunger = 0.0
        self.sleepiness = 0.0
        self.age = 0
        self.lifespan = lifespan  # Срок жизни
        
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
        self.age += 1
        
    def is_alive(self):
        return self.energy > 0 and self.age < self.lifespan

class EvolvingBrain:
    def __init__(self, n_sensors=9, n_actions=1):
        # 9 сенсоров: temp, pain, fear, energy, hunger, sleepiness, 
        #             боль партнера, возраст, шум
        self.pred_brain = PredictiveBrain(n_sensors=n_sensors, n_actions=n_actions, lr=0.1, lr_action=0.1)
        self.trauma_memory = []
        self.seen_states = set()
        self.age = 0
        
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

class DigitalEntity:
    def __init__(self, name, brain, lifespan=500):
        self.name = name
        self.brain = brain
        self.body = VirtualBody(lifespan=lifespan)
        self.brain.age = 0
        
    def act(self, sensors):
        return self.brain.act(sensors, np.array([0.5]), self.body.pain, self.body.temp, self.body.hunger, self.body.sleepiness)
    
    def learn(self, sensors, action, next_sensors):
        self.brain.learn(sensors, action, next_sensors, self.body.pain)
    
    def update_body(self, action):
        self.body.update(action)
        self.brain.age += 1
    
    def is_alive(self):
        return self.body.is_alive()
    
    def consolidate(self):
        self.brain.consolidate_memory()

def evaluate_entity(entity, steps=100):
    """Оцениваем одного агента в изоляции (но с виртуальным партнером)"""
    body = entity.body
    brain = entity.brain
    total_pain = 0.0
    total_fear = 0.0
    total_curiosity = 0.0
    
    # Виртуальный партнер (с фиксированной болью)
    partner_pain = 0.5
    
    for step in range(steps):
        # Ночь каждые 20 шагов
        if step % 20 == 0 and step > 0:
            brain.consolidate_memory()
            body.energy = min(1.0, body.energy + 0.05)
            body.sleepiness = max(0.0, body.sleepiness - 0.5)
            continue
        
        sensors = np.array([body.temp, body.pain, body.fear, body.energy, 
                            body.hunger, body.sleepiness, partner_pain, body.age / body.lifespan, 0.5])
        
        action, fear, curiosity = entity.act(sensors)
        body.fear = fear
        
        if body.pain < 0.1:
            body.fear = max(0.0, body.fear - 0.1)
        
        entity.update_body(action)
        
        total_pain += body.pain
        total_fear += body.fear
        total_curiosity += curiosity
        
        body.energy -= 0.005 + body.pain * 0.02 + body.hunger * 0.01
        
        if 0.4 <= body.temp <= 0.6:
            body.energy = min(1.0, body.energy + 0.01)
        
        next_sensors = np.array([body.temp, body.pain, body.fear, body.energy, 
                                 body.hunger, body.sleepiness, partner_pain, body.age / body.lifespan, 0.5])
        entity.learn(sensors, action, next_sensors)
        
        if not entity.is_alive():
            break
    
    fitness = (step + 1) - (total_pain / max(1, steps)) * 100 - (total_fear / max(1, steps)) * 20 + total_curiosity * 10
    return fitness, step + 1, body.energy, total_pain

def evolve_population():
    population = []
    
    # Загружаем ген, если есть
    if os.path.exists(GENE_FILE):
        with open(GENE_FILE, 'rb') as f:
            data = pickle.load(f)
        for i in range(3):
            brain = EvolvingBrain(n_sensors=9, n_actions=1)
            brain.pred_brain.W_s = data['W_s']
            brain.pred_brain.W_a = data['W_a']
            brain.trauma_memory = data.get('trauma_memory', [])
            brain.pred_brain.W_s += np.random.normal(0, 0.1, brain.pred_brain.W_s.shape)
            brain.pred_brain.W_a += np.random.normal(0, 0.1, brain.pred_brain.W_a.shape)
            population.append(DigitalEntity(f"Эон-{i}", brain))
    
    while len(population) < 10:
        brain = EvolvingBrain(n_sensors=9, n_actions=1)
        brain.pred_brain.W_a += np.random.normal(0, 0.2, brain.pred_brain.W_a.shape)
        population.append(DigitalEntity(f"Эон-{len(population)}", brain))
    
    best_overall = None
    best_fitness = -999
    
    for gen in range(20):
        results = []
        for entity in population:
            fitness, life, energy, pain = evaluate_entity(entity, steps=100)
            results.append((fitness, life, entity, energy, pain))
        
        results.sort(key=lambda x: x[0], reverse=True)
        best = results[0]
        
        if best[0] > best_fitness:
            best_fitness = best[0]
            best_overall = copy.deepcopy(best[2])
            # Сохраняем ген
            with open(GENE_FILE, 'wb') as f:
                pickle.dump({
                    'W_s': best_overall.brain.pred_brain.W_s,
                    'W_a': best_overall.brain.pred_brain.W_a,
                    'trauma_memory': best_overall.brain.trauma_memory
                }, f)
            print(f"  Поколение {gen}: Fitness={best[0]:.1f}, Жизнь={best[1]}, Боль={best[4]:.2f} <-- ЛУЧШИЙ (ген сохранен)")
        else:
            print(f"  Поколение {gen}: Fitness={best[0]:.1f}, Жизнь={best[1]}, Боль={best[4]:.2f}")
        
        # Размножение: топ-3 дают потомков
        new_population = []
        top3 = [results[0][2], results[1][2], results[2][2]]
        
        for i in range(10):
            parent1 = top3[np.random.randint(0, 3)]
            parent2 = top3[np.random.randint(0, 3)]
            
            # Кроссовер
            child_brain = EvolvingBrain(n_sensors=9, n_actions=1)
            mask = np.random.rand(*parent1.brain.pred_brain.W_s.shape) < 0.5
            child_brain.pred_brain.W_s = np.where(mask, parent1.brain.pred_brain.W_s, parent2.brain.pred_brain.W_s)
            mask_a = np.random.rand(*parent1.brain.pred_brain.W_a.shape) < 0.5
            child_brain.pred_brain.W_a = np.where(mask_a, parent1.brain.pred_brain.W_a, parent2.brain.pred_brain.W_a)
            
            # Мутации
            child_brain.pred_brain.W_s += np.random.normal(0, 0.1, child_brain.pred_brain.W_s.shape)
            child_brain.pred_brain.W_a += np.random.normal(0, 0.1, child_brain.pred_brain.W_a.shape)
            
            # Наследуем травмы
            child_brain.trauma_memory = copy.deepcopy(parent1.brain.trauma_memory)
            
            child = DigitalEntity(f"Эон-{gen}-{i}", child_brain, lifespan=500)
            new_population.append(child)
        
        population = new_population
    
    return best_overall

def main():
    print("=== РАЗМНОЖЕНИЕ И СМЕРТЬ (20 ПОКОЛЕНИЙ) ===\n")
    best_entity = evolve_population()
    
    print("\n=== ФИНАЛЬНЫЙ ТЕСТ ЛУЧШЕГО АГЕНТА ===")
    fitness, life, energy, pain = evaluate_entity(best_entity, steps=200)
    print(f"Прожито тактов: {life}")
    print(f"Остаток энергии: {energy:.3f}")
    print(f"Суммарная боль: {pain:.2f}")
    print(f"Fitness: {fitness:.1f}")

if __name__ == "__main__":
    main()
