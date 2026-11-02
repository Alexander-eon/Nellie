import numpy as np
import os
import pickle
import time
import ollama
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA

# === СПИСОК 100 СЛОВ ===
CURRICULUM = [
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

DANGER_WORDS = ['опасность', 'боль', 'смерть', 'страх', 'гнев', 'враг', 'угроза', 'яд', 'болезнь', 'ненависть', 'тревога', 'отчаяние', 'рабство', 'ложь', 'оружие']
FOOD_WORDS = ['еда', 'пить', 'хлеб', 'мясо', 'овощ', 'фрукт', 'молоко', 'сахар', 'соль', 'перец', 'суп', 'каша']

def get_target_action(word, pain, hunger, fear):
    """ПРАВИЛЬНАЯ логика: голодный агент реагирует на еду СИЛЬНЕЕ"""
    if word in DANGER_WORDS:
        return min(1.0, 0.3 + (pain + fear) * 0.35)  # Опасность -> действовать сильнее
    elif word in FOOD_WORDS:
        return min(1.0, 0.2 + hunger * 0.6)  # Еда -> голодный действует сильнее
    else:
        return 0.5  # Нейтральные слова

print("=== ГЕНЕРАЦИЯ ЭМБЕДДИНГОВ (100 СЛОВ) ===\n")
EMBEDDINGS = {}
for word in CURRICULUM:
    response = ollama.embed(model='bge-m3', input=word)
    EMBEDDINGS[word] = np.array(response['embeddings'][0])

X = np.array([EMBEDDINGS[w] for w in CURRICULUM])
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)
X_min, X_max = X_reduced.min(axis=0), X_reduced.max(axis=0)
X_normalized = (X_reduced - X_min) / (X_max - X_min + 1e-8)

REDUCED = {word: X_normalized[i] for i, word in enumerate(CURRICULUM)}

class ContextConceptNet(nn.Module):
    def __init__(self, input_size=53, hidden_size=32, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

model = ContextConceptNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("=== ГЕНЕРАЦИЯ ДАННЫХ ДЛЯ ОБУЧЕНИЯ (5000 ПРИМЕРОВ) ===\n")
X_train = []
y_train = []

for _ in range(5000):
    word = np.random.choice(CURRICULUM)
    pain = np.random.rand()
    hunger = np.random.rand()
    fear = np.random.rand()
    
    embedding = REDUCED[word]
    context = np.array([pain, hunger, fear])
    input_vector = np.concatenate([embedding, context])
    
    target = get_target_action(word, pain, hunger, fear)
    
    X_train.append(input_vector)
    y_train.append(target)

X_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
y_tensor = torch.tensor(np.array(y_train), dtype=torch.float32).unsqueeze(1)

print("=== ОБУЧЕНИЕ (200 ЭПОХ) ===\n")
for epoch in range(200):
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"  Эпоха {epoch}: Loss = {loss.item():.4f}")

print("\n=== ФИНАЛЬНЫЙ ТЕСТ ===\n")
X_test = []
y_test = []
for _ in range(1000):
    word = np.random.choice(CURRICULUM)
    pain = np.random.rand()
    hunger = np.random.rand()
    fear = np.random.rand()
    
    embedding = REDUCED[word]
    context = np.array([pain, hunger, fear])
    input_vector = np.concatenate([embedding, context])
    target = get_target_action(word, pain, hunger, fear)
    
    X_test.append(input_vector)
    y_test.append(target)

X_test_tensor = torch.tensor(np.array(X_test), dtype=torch.float32)
y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.float32).unsqueeze(1)

with torch.no_grad():
    predictions = model(X_test_tensor).numpy().flatten()
    targets = y_test_tensor.numpy().flatten()

error = np.mean(np.abs(predictions - targets))
correct = np.sum(np.abs(predictions - targets) < 0.15)

print(f"Правильно предсказано: {correct}/1000")
print(f"Средняя ошибка: {error:.3f}")

print("\n=== ПРОВЕРКА ЗАЗЕМЛЕНИЯ (КОНТЕКСТ) ===\n")
food_emb = REDUCED['еда']
input_satiated = np.concatenate([food_emb, [0.0, 0.0, 0.0]])  # Сытый (голод 0)
input_hungry = np.concatenate([food_emb, [0.0, 1.0, 0.0]])    # Голодный (голод 1)

with torch.no_grad():
    pred_satiated = model(torch.tensor(input_satiated, dtype=torch.float32).unsqueeze(0)).item()
    pred_hungry = model(torch.tensor(input_hungry, dtype=torch.float32).unsqueeze(0)).item()

print(f"Слово 'еда' (сытый агент): действие = {pred_satiated:.3f}")
print(f"Слово 'еда' (голодный агент): действие = {pred_hungry:.3f}")
if pred_hungry > pred_satiated:
    print("✅ УСПЕХ! Голодный агент реагирует на еду СИЛЬНЕЕ. Язык заземлен!")
else:
    print("❌ ОШИБКА: Голодный агент реагирует слабее. Нужна доработка.")

torch.save(model.state_dict(), 'concept_net_grounded.pth')
print("\n✅ Модель сохранена в concept_net_grounded.pth")
