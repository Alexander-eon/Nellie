import numpy as np
import copy
import pickle
import os
from predictive_brain import PredictiveBrain

GENE_FILE = "digital_gene_comm.pkl"

class VirtualBody:
    def __init__(self, initial_temp=0.1):
        self.temp = initial_temp
        self.pain = 0.0
        self.fear = 0.0
        self.energy = 1.0
        
    def update(self, action):
        heat = (action - 0.5) * 0.3
        self.temp = np.clip(self.temp + heat, 0.0, 1.0)
        if self.temp > 0.7:
            self.pain = (self.temp - 0.7) / 0.3
        elif self.temp < 0.3:
            self.pain = (0.3 - self.temp) / 0.3
        else:
            self.pain = 0.0

class CommunicativeBrain:
    def __init__(self, n_sensors=6, n_actions=1):
        # 6 сенсоров: свой temp, свой pain, свой fear, своя энергия, 
        #             сообщение партнера (его боль), шум
        # 1 действие: физическое (нагрев/охлаждение)
        # Сообщение = производное от своего pain (не действие)
        self.pred_brain = PredictiveBrain(n_sensors=n_sensors, n_actions=n_actions, lr=0.1, lr_action=0.1)
        self.trauma_memory = []
        self.seen_states = set()
        
    def act(self, sensors, target, pain_level, current_temp):
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
        
        if fear > 0.3:
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

def evaluate_pair(brains, steps=100):
    brain_a, brain_b = brains
    body_a = VirtualBody(initial_temp=0.1)
    body_b = VirtualBody(initial_temp=0.8)
    
    total_pain_a, total_pain_b = 0.0, 0.0
    total_fear_a, total_curiosity_a = 0.0, 0.0
    messages_received = 0
    
    last_msg_b = 0.5  # Сообщение B (его боль)
    last_msg_a = 0.5  # Сообщение A (его боль)
    
    for step in range(steps):
        # А видит: свой temp, свой pain, свой fear, свою энергию, боль B, шум
        sensors_a = np.array([body_a.temp, body_a.pain, body_a.fear, body_a.energy, last_msg_b, 0.5])
        # B видит: свой temp, свой pain, свой fear, свою энергию, боль A, шум
        sensors_b = np.array([body_b.temp, body_b.pain, body_b.fear, body_b.energy, last_msg_a, 0.5])
        
        target_a = np.array([0.5])
        target_b = np.array([0.5])
        
        action_a, fear_a, curiosity_a = brain_a.act(sensors_a, target_a, body_a.pain, body_a.temp)
        action_b, fear_b, curiosity_b = brain_b.act(sensors_b, target_b, body_b.pain, body_b.temp)
        
        body_a.fear = fear_a
        body_b.fear = fear_b
        
        if body_a.pain < 0.1:
            body_a.fear = max(0.0, body_a.fear - 0.1)
        if body_b.pain < 0.1:
            body_b.fear = max(0.0, body_b.fear - 0.1)
        
        body_a.update(action_a)
        body_b.update(action_b)
        
        total_pain_a += body_a.pain
        total_pain_b += body_b.pain
        total_fear_a += body_a.fear
        total_curiosity_a += curiosity_a
        
        # СООБЩЕНИЕ = боль агента
        last_msg_a = body_a.pain
        last_msg_b = body_b.pain
        
        # Бонус за полезное сообщение (если сообщение коррелирует с болью)
        if abs(body_a.pain - last_msg_b) < 0.2:
            messages_received += 1
        if abs(body_b.pain - last_msg_a) < 0.2:
            messages_received += 1
        
        body_a.energy -= 0.005 + body_a.pain * 0.02
        body_b.energy -= 0.005 + body_b.pain * 0.02
        
        if 0.4 <= body_a.temp <= 0.6:
            body_a.energy = min(1.0, body_a.energy + 0.01)
        if 0.4 <= body_b.temp <= 0.6:
            body_b.energy = min(1.0, body_b.energy + 0.01)
        
        next_sensors_a = np.array([body_a.temp, body_a.pain, body_a.fear, body_a.energy, body_b.pain, 0.5])
        next_sensors_b = np.array([body_b.temp, body_b.pain, body_b.fear, body_b.energy, body_a.pain, 0.5])
        brain_a.learn(sensors_a, action_a, next_sensors_a, body_a.pain)
        brain_b.learn(sensors_b, action_b, next_sensors_b, body_b.pain)
        
        if body_a.energy <= 0 or body_b.energy <= 0:
            break
    
    fitness_a = (step + 1) - (total_pain_a / max(1, steps)) * 100 - (total_fear_a / max(1, steps)) * 20 + total_curiosity_a * 10 + messages_received * 5
    fitness_b = (step + 1) - (total_pain_b / max(1, steps)) * 100
    fitness = (fitness_a + fitness_b) / 2
    
    return fitness, step + 1, body_a.energy, body_b.energy, total_pain_a, total_pain_b, total_curiosity_a, messages_received

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
        brain = CommunicativeBrain(n_sensors=6, n_actions=1)
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
            brain = CommunicativeBrain(n_sensors=6, n_actions=1)
        population.append(brain)
    
    for i in range(7):
        brain = CommunicativeBrain(n_sensors=6, n_actions=1)
        brain.pred_brain.W_a += np.random.normal(0, 0.2, brain.pred_brain.W_a.shape)
        population.append(brain)
    
    best_overall = None
    best_fitness = -999
    
    for gen in range(20):
        results = []
        pairs = [(population[i], population[(i+1) % len(population)]) for i in range(len(population))]
        
        for brain_a, brain_b in pairs:
            fitness, life, energy_a, energy_b, pain_a, pain_b, curiosity, received = evaluate_pair((brain_a, brain_b), steps=50)
            results.append((fitness, life, brain_a, brain_b, energy_a, energy_b, pain_a, pain_b, curiosity, received))
        
        results.sort(key=lambda x: x[0], reverse=True)
        best = results[0]
        
        if best[0] > best_fitness:
            best_fitness = best[0]
            best_overall = copy.deepcopy(best[2])
            save_gene(best_overall)
            print(f"  Поколение {gen}: Fitness={best[0]:.1f}, Боль A={best[6]:.2f}, Любопытство={best[8]:.2f}, Полезных={best[9]} <-- ЛУЧШИЙ")
        else:
            print(f"  Поколение {gen}: Fitness={best[0]:.1f}, Боль A={best[6]:.2f}, Любопытство={best[8]:.2f}, Полезных={best[9]}")
        
        new_population = []
        top3 = [results[0][2], results[1][2], results[2][2]]
        for i in range(10):
            parent1 = top3[np.random.randint(0, 3)]
            parent2 = top3[np.random.randint(0, 3)]
            child = CommunicativeBrain(n_sensors=6, n_actions=1)
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
    print("=== ОБЩЕНИЕ (20 ПОКОЛЕНИЙ) ===\n")
    best_brain = evolve_population()
    print("\n=== ФИНАЛЬНЫЙ ТЕСТ ===")
    partner = copy.deepcopy(best_brain)
    fitness, life, energy_a, energy_b, pain_a, pain_b, curiosity, received = evaluate_pair((best_brain, partner), steps=100)
    print(f"Прожито тактов: {life}")
    print(f"Энергия A: {energy_a:.3f}, Энергия B: {energy_b:.3f}")
    print(f"Боль A: {pain_a:.2f}, Боль B: {pain_b:.2f}")
    print(f"Любопытство: {curiosity:.2f}")
    print(f"Полезных сообщений: {received}")
    print(f"Fitness: {fitness:.1f}")

if __name__ == "__main__":
    main()
