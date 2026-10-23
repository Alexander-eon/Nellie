import numpy as np
import copy
import pickle
import os
from predictive_brain import PredictiveBrain

GENE_FILE = "digital_gene.pkl"

class VirtualBody:
    def __init__(self, initial_temp=0.1):
        self.temp = initial_temp
        self.pain = 0.0
        self.fear = 0.0
        
    def update(self, action):
        heat = (action - 0.5) * 0.3
        self.temp = np.clip(self.temp + heat, 0.0, 1.0)
        
        if self.temp > 0.7:
            self.pain = (self.temp - 0.7) / 0.3
        elif self.temp < 0.3:
            self.pain = (0.3 - self.temp) / 0.3
        else:
            self.pain = 0.0

class FearBrain:
    def __init__(self, n_sensors=5, n_actions=1):
        self.pred_brain = PredictiveBrain(n_sensors=n_sensors, n_actions=n_actions, lr=0.1, lr_action=0.1)
        self.trauma_memory = []
        
    def act(self, sensors, target, pain_level, current_temp):
        # Считаем страх: ищем похожие травмы
        fear = 0.0
        current_state = tuple(np.round([current_temp, sensors[1], sensors[2]], 1))
        
        for trauma in self.trauma_memory:
            distance = sum(abs(a - b) for a, b in zip(current_state, trauma['state']))
            if distance < 0.3:
                fear = max(fear, trauma['pain'])
        
        # СТРАХ ПОМОГАЕТ: если страх высокий, агент действует агрессивнее
        # Сдвигаем цель: если страх, то цель не 0.5, а 0.3 (хотим больше тепла, чтобы не замерзнуть)
        # или 0.7 (хотим меньше тепла, чтобы не перегреться)
        if fear > 0.3:
            # Если текущая температура низкая, страх "толкает" к нагреву
            if current_temp < 0.5:
                target = np.array([0.6 + fear * 0.2])  # Нагреваемся
            else:
                target = np.array([0.4 - fear * 0.2])  # Охлаждаемся
        
        action = self.pred_brain.act(sensors, target)
        return np.clip(action[0], 0.0, 1.0), fear
        
    def learn(self, sensors, action, next_sensors, pain_level):
        self.pred_brain.learn(sensors, np.array([action]), next_sensors)
        
        # Запоминаем травму, только если боль была ОЧЕНЬ сильной
        if pain_level > 0.8:  # Жесткий порог (раньше был 0.5)
            state = tuple(np.round(sensors[:3], 1))
            # Не дублируем травмы
            for trauma in self.trauma_memory:
                if trauma['state'] == state:
                    trauma['pain'] = max(trauma['pain'], pain_level)
                    return
            self.trauma_memory.append({'state': state, 'pain': pain_level})
            if len(self.trauma_memory) > 20:  # Меньше памяти (раньше было 100)
                self.trauma_memory.pop(0)

def evaluate_agent(brain, steps=100, initial_temp=0.1):
    body = VirtualBody(initial_temp=initial_temp)
    energy = 1.0
    total_pain = 0.0
    total_fear = 0.0
    
    for step in range(steps):
        sensors = np.array([body.temp, body.pain, body.fear, energy, 0.5])
        target = np.array([0.5])
        
        action, fear = brain.act(sensors, target, body.pain, body.temp)
        body.fear = fear
        
        # Страх уменьшается, если агент избежал боли
        if body.pain < 0.1:
            body.fear = max(0.0, body.fear - 0.1)  # Страх спадает
        
        body.update(action)
        
        total_pain += body.pain
        total_fear += body.fear
        energy -= 0.005
        energy -= body.pain * 0.02
        energy -= body.fear * 0.005  # Страх почти не отнимает энергию
        
        if 0.4 <= body.temp <= 0.6:
            energy = min(1.0, energy + 0.01)
        
        next_sensors = np.array([body.temp, body.pain, body.fear, energy, 0.5])
        brain.learn(sensors, action, next_sensors, body.pain)
        
        if energy <= 0:
            break
    
    fitness = (step + 1) - (total_pain / max(1, steps)) * 100 - (total_fear / max(1, steps)) * 20
    return fitness, step + 1, energy, total_pain, total_fear, len(brain.trauma_memory)

def evolve_population():
    population = []
    
    # Загружаем ген, если есть
    if os.path.exists(GENE_FILE):
        with open(GENE_FILE, 'rb') as f:
            data = pickle.load(f)
        for i in range(3):
            brain = FearBrain(n_sensors=5, n_actions=1)
            # Загружаем только если размерности совпадают
            if data['W_s'].shape[0] == 5:
                brain.pred_brain.W_s = data['W_s']
                brain.pred_brain.W_a = data['W_a']
                brain.pred_brain.W_s += np.random.normal(0, 0.1, brain.pred_brain.W_s.shape)
                brain.pred_brain.W_a += np.random.normal(0, 0.1, brain.pred_brain.W_a.shape)
            population.append(brain)
    
    while len(population) < 10:
        brain = FearBrain(n_sensors=5, n_actions=1)
        brain.pred_brain.W_a += np.random.normal(0, 0.2, brain.pred_brain.W_a.shape)
        population.append(brain)
    
    best_overall = None
    best_fitness = -999
    
    for gen in range(30):
        results = []
        for brain in population:
            # Обучаем на 3 эпизодах (чтобы накопить опыт)
            for episode in range(3):
                fitness, life, energy, pain, fear, traumas = evaluate_agent(brain, steps=50, initial_temp=np.random.uniform(0.0, 0.3))
            results.append((fitness, life, brain, energy, pain, fear, traumas))
        
        results.sort(key=lambda x: x[0], reverse=True)
        best = results[0]
        
        if best[0] > best_fitness:
            best_fitness = best[0]
            best_overall = copy.deepcopy(best[2])
            print(f"  Поколение {gen}: Fitness={best[0]:.1f}, Боль={best[4]:.2f}, Страх={best[5]:.2f}, Травм={best[6]}  <-- ЛУЧШИЙ")
        else:
            print(f"  Поколение {gen}: Fitness={best[0]:.1f}, Боль={best[4]:.2f}, Страх={best[5]:.2f}, Травм={best[6]}")
        
        # Кроссовер
        new_population = []
        top3 = [results[0][2], results[1][2], results[2][2]]
        
        for i in range(10):
            parent1 = top3[np.random.randint(0, 3)]
            parent2 = top3[np.random.randint(0, 3)]
            
            child = FearBrain(n_sensors=5, n_actions=1)
            mask = np.random.rand(*parent1.pred_brain.W_s.shape) < 0.5
            child.pred_brain.W_s = np.where(mask, parent1.pred_brain.W_s, parent2.pred_brain.W_s)
            mask_a = np.random.rand(*parent1.pred_brain.W_a.shape) < 0.5
            child.pred_brain.W_a = np.where(mask_a, parent1.pred_brain.W_a, parent2.pred_brain.W_a)
            
            child.pred_brain.W_s += np.random.normal(0, 0.1, child.pred_brain.W_s.shape)
            child.pred_brain.W_a += np.random.normal(0, 0.1, child.pred_brain.W_a.shape)
            
            # Наследуем травмы
            child.trauma_memory = copy.deepcopy(parent1.trauma_memory)
            new_population.append(child)
        
        population = new_population
    
    return best_overall

def main():
    print("=== ЭВОЛЮЦИЯ СТРАХА (АДАПТИВНЫЙ СТРАХ) ===\n")
    
    best_brain = evolve_population()
    
    print("\n=== ФИНАЛЬНЫЙ ТЕСТ ===")
    fitness, life, energy, pain, fear, traumas = evaluate_agent(best_brain, steps=100, initial_temp=0.1)
    print(f"Прожито тактов: {life}")
    print(f"Остаток энергии: {energy:.3f}")
    print(f"Суммарная боль: {pain:.2f}")
    print(f"Суммарный страх: {fear:.2f}")
    print(f"Травм в памяти: {traumas}")
    print(f"Fitness: {fitness:.1f}")

if __name__ == "__main__":
    main()
