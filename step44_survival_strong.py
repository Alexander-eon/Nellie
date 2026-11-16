import numpy as np
import os
import pickle
import time
import ollama
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

# === ЗАГРУЗКА МОДЕЛИ ===
class StrongDualNet(nn.Module):
    def __init__(self, embed_size=50, context_size=3, hidden=16):
        super().__init__()
        self.fc_embed = nn.Linear(embed_size, hidden * 2)
        self.tanh_embed = nn.Tanh()
        self.fc_context = nn.Linear(context_size, hidden // 2)
        self.tanh_context = nn.Tanh()
        self.fc_merge = nn.Linear(hidden * 2 + hidden // 2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embed = x[:, :50]
        context = x[:, 50:]
        h_embed = self.tanh_embed(self.fc_embed(embed))
        h_context = self.tanh_context(self.fc_context(context))
        merged = torch.cat([h_embed, h_context], dim=1)
        out = self.sigmoid(self.fc_merge(merged))
        return out

model = StrongDualNet()
model.load_state_dict(torch.load('strong_dual_net.pth'))
model.eval()
print("✅ Модель StrongDualNet загружена.\n")

# === ЗАГРУЗКА ЭМБЕДДИНГОВ ===
print("=== ЗАГРУЗКА ЭМБЕДДИНГОВ ===\n")
WORDS = [
    'опасность', 'еда', 'боль', 'дом', 'бежать', 'спать', 'пить', 'друг', 'враг', 'огонь',
    'собака', 'кошка', 'волк', 'идти', 'стоять', 'вода', 'земля', 'воздух', 'солнце', 'луна',
    'радость', 'страх', 'гнев', 'машина', 'оружие', 'человек', 'ребенок', 'женщина', 'мужчина', 'учитель',
    'дерево', 'цветок', 'трава', 'камень', 'хлеб', 'мясо', 'овощ', 'фрукт', 'молоко', 'сахар',
    'сидеть', 'лежать', 'есть', 'говорить', 'слушать', 'любовь', 'ненависть', 'спокойствие', 'тревога', 'надежда',
    'стол', 'стул', 'книга', 'компьютер', 'телефон', 'инструмент', 'одежда', 'время', 'пространство', 'жизнь',
    'смерть', 'правда', 'ложь', 'свобода', 'рабство', 'власть', 'деньги', 'большой', 'маленький', 'быстрый',
    'медленный', 'горячий', 'холодный', 'светлый', 'темный', 'тяжелый', 'легкий', 'соль', 'перец', 'суп',
    'каша', 'здоровье', 'болезнь', 'лекарство', 'яд', 'защита', 'угроза', 'помощь', 'безопасность', 'птица',
    'рыба', 'лошадь', 'корова', 'свинья', 'курица', 'мышь', 'отчаяние', 'царь', 'воин', 'врач'
]

EMBEDDINGS = {}
for word in WORDS:
    response = ollama.embed(model='bge-m3', input=word)
    EMBEDDINGS[word] = np.array(response['embeddings'][0])

X = np.array([EMBEDDINGS[w] for w in WORDS])
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)
X_min, X_max = X_reduced.min(axis=0), X_reduced.max(axis=0)
X_normalized = (X_reduced - X_min) / (X_max - X_min + 1e-8)
REDUCED = {word: X_normalized[i] for i, word in enumerate(WORDS)}
print(f"✅ Загружено {len(REDUCED)} эмбеддингов.\n")

# === ВИРТУАЛЬНЫЙ МИР ===
class VirtualWorld:
    def __init__(self):
        self.energy = 1.0
        self.pain = 0.0
        self.hunger = 0.0
        self.fear = 0.0
        self.step = 0
        self.alive = True
        
    def update(self, action, word):
        """Обновляем мир на основе действия агента и слова"""
        if word in ['еда', 'хлеб', 'мясо', 'овощ', 'фрукт', 'молоко', 'сахар', 'суп', 'каша']:
            # Еда: чем выше действие, тем быстрее утоляется голод
            self.hunger = max(0.0, self.hunger - action * 0.5)
            self.energy = min(1.0, self.energy + action * 0.1)
        elif word in ['опасность', 'боль', 'смерть', 'страх', 'гнев', 'враг', 'угроза', 'яд', 'болезнь', 'ненависть', 'тревога', 'отчаяние', 'оружие']:
            # Опасность: чем выше действие, тем сильнее убегаем (страх падает, энергия тратится)
            self.fear = max(0.0, self.fear - action * 0.5)
            self.energy -= action * 0.05
        else:
            # Нейтральное слово: просто тратим энергию
            self.energy -= 0.01
        
        # Голод и боль растут со временем
        self.hunger = min(1.0, self.hunger + 0.02)
        self.pain = min(1.0, self.pain + 0.01)
        
        # Если энергия кончилась — смерть
        if self.energy <= 0:
            self.alive = False
        
        self.step += 1
        return self.energy, self.alive

# === АГЕНТ ===
def agent_act(word, energy, pain, hunger, fear):
    """Агент действует на основе слова и контекста"""
    embedding = REDUCED[word]
    context = np.array([pain, hunger, fear])
    input_vector = np.concatenate([embedding, context])
    
    with torch.no_grad():
        action = model(torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)).item()
    
    return action

# === ЗАПУСК ВЫЖИВАНИЯ ===
print("=== ВИРТУАЛЬНЫЙ МИР: СИЛЬНАЯ ДИФФЕРЕНЦИАЦИЯ ===\n")

world = VirtualWorld()
world.hunger = 0.8
world.fear = 0.7

events = [
    'еда', 'опасность', 'еда', 'волк', 'хлеб', 'боль',
    'мясо', 'огонь', 'вода', 'смерть', 'фрукт', 'враг',
    'суп', 'угроза', 'каша', 'яд', 'молоко', 'оружие'
]

print("Шаг | Слово | Действие | Энергия | Голод | Страх")
print("-" * 60)

for step in range(100):
    word = np.random.choice(events)
    action = agent_act(word, world.energy, world.pain, world.hunger, world.fear)
    energy, alive = world.update(action, word)
    
    if step % 10 == 0:
        print(f"{step:3d} | {word:10} | {action:.2f} | {energy:.2f} | {world.hunger:.2f} | {world.fear:.2f}")
    
    if not alive:
        print(f"☠️ Агент умер на шаге {step}")
        break

print(f"\n=== ИТОГИ ===")
print(f"Прожито тактов: {step + 1}")
print(f"Остаток энергии: {world.energy:.2f}")
print(f"Голод: {world.hunger:.2f}")
print(f"Страх: {world.fear:.2f}")

# === ПРОВЕРКА РЕАКЦИЙ ===
print("\n=== ПРОВЕРКА РЕАКЦИЙ НА СЛОВА ===\n")
test_words = ['еда', 'опасность', 'вода', 'волк', 'хлеб', 'огонь']
for word in test_words:
    action = agent_act(word, 0.5, 0.5, 0.8, 0.8)
    print(f"  {word:10}: действие = {action:.2f}")
