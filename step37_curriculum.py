import numpy as np
import os
import pickle
import time
import ollama
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# === БАЗОВЫЙ СПИСОК (100 СЛОВ, ОТСОРТИРОВАННЫХ ПО СЛОЖНОСТИ) ===
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

ACTIONS = {
    'опасность': 0.9, 'боль': 0.9, 'смерть': 0.9, 'страх': 0.9, 'гнев': 0.9,
    'враг': 0.8, 'угроза': 0.9, 'яд': 0.9, 'болезнь': 0.8, 'ненависть': 0.8,
    'тревога': 0.8, 'отчаяние': 0.8, 'рабство': 0.8, 'ложь': 0.7, 'оружие': 0.8,
    'еда': 0.2, 'пить': 0.2, 'хлеб': 0.2, 'мясо': 0.2, 'овощ': 0.2, 'фрукт': 0.2,
    'молоко': 0.2, 'сахар': 0.2, 'соль': 0.2, 'перец': 0.2, 'суп': 0.2, 'каша': 0.2,
    'радость': 0.1, 'любовь': 0.1, 'спокойствие': 0.1, 'надежда': 0.1, 'друг': 0.1,
    'безопасность': 0.1, 'здоровье': 0.1, 'лекарство': 0.2, 'защита': 0.2, 'помощь': 0.2,
    'собака': 0.5, 'кошка': 0.5, 'птица': 0.5, 'рыба': 0.5, 'лошадь': 0.5,
    'корова': 0.5, 'свинья': 0.5, 'курица': 0.5, 'мышь': 0.5, 'волк': 0.5,
    'дерево': 0.5, 'цветок': 0.5, 'трава': 0.5, 'камень': 0.5, 'вода': 0.5,
    'огонь': 0.6, 'земля': 0.5, 'воздух': 0.5, 'солнце': 0.5, 'луна': 0.5,
    'человек': 0.5, 'ребенок': 0.5, 'женщина': 0.5, 'мужчина': 0.5,
    'учитель': 0.5, 'врач': 0.5, 'воин': 0.6, 'царь': 0.5,
    'грусть': 0.6, 'бежать': 0.6, 'идти': 0.5, 'стоять': 0.5, 'сидеть': 0.5,
    'лежать': 0.5, 'есть': 0.5, 'спать': 0.5, 'говорить': 0.5, 'слушать': 0.5,
    'машина': 0.5, 'дом': 0.5, 'стол': 0.5, 'стул': 0.5, 'книга': 0.5,
    'компьютер': 0.5, 'телефон': 0.5, 'инструмент': 0.5, 'одежда': 0.5,
    'время': 0.5, 'пространство': 0.5, 'жизнь': 0.5, 'правда': 0.5,
    'свобода': 0.5, 'власть': 0.6, 'деньги': 0.5,
    'большой': 0.5, 'маленький': 0.5, 'быстрый': 0.5, 'медленный': 0.5,
    'горячий': 0.5, 'холодный': 0.5, 'светлый': 0.5, 'темный': 0.5,
    'тяжелый': 0.5, 'легкий': 0.5,
}

print("=== ГЕНЕРАЦИЯ ЭМБЕДДИНГОВ (100 СЛОВ) ===\n")
EMBEDDINGS = {}
for word in CURRICULUM:
    response = ollama.embed(model='bge-m3', input=word)
    EMBEDDINGS[word] = np.array(response['embeddings'][0])

print("=== PCA (СЖАТИЕ ДО 50) ===\n")
X = np.array([EMBEDDINGS[w] for w in CURRICULUM])
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)
X_min, X_max = X_reduced.min(axis=0), X_reduced.max(axis=0)
X_normalized = (X_reduced - X_min) / (X_max - X_min + 1e-8)

REDUCED = {word: X_normalized[i] for i, word in enumerate(CURRICULUM)}

class ConceptNet(nn.Module):
    def __init__(self, input_size=50, hidden_size=32, output_size=1):
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

print("=== ПОЭТАПНОЕ ОБУЧЕНИЕ (10 -> 100 СЛОВ) ===\n")

model = ConceptNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

X_all = torch.tensor(np.array([REDUCED[w] for w in CURRICULUM]), dtype=torch.float32)
y_all = torch.tensor([ACTIONS[w] for w in CURRICULUM], dtype=torch.float32).unsqueeze(1)

def get_stage_mask(stage):
    words_so_far = (stage + 1) * 10
    mask = torch.zeros(len(CURRICULUM), dtype=torch.bool)
    mask[:words_so_far] = True
    return mask

results = []

for stage in range(10):
    mask = get_stage_mask(stage)
    X_stage = X_all[mask]
    y_stage = y_all[mask]
    
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(X_stage)
        loss = criterion(output, y_stage)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        predictions = model(X_stage).squeeze().numpy()
    
    correct = 0
    for i in range(len(predictions)):
        true_action = y_stage[i].item()
        if abs(true_action - predictions[i]) < 0.15:
            correct += 1
    
    accuracy = correct / len(predictions) * 100
    results.append(((stage + 1) * 10, accuracy, loss.item()))
    print(f"  Этап {stage + 1} ({len(predictions)} слов): Точность {accuracy:.1f}%, Loss {loss.item():.4f}")

print("\n=== ФИНАЛЬНЫЙ ТЕСТ НА ВСЕХ 100 СЛОВАХ ===\n")
with torch.no_grad():
    predictions = model(X_all).squeeze().numpy()

correct = 0
for i, word in enumerate(CURRICULUM):
    true_action = ACTIONS[word]
    if abs(true_action - predictions[i]) < 0.15:
        correct += 1

print(f"Правильно предсказано: {correct}/100")
print(f"Средняя ошибка: {np.mean(np.abs(predictions - np.array([ACTIONS[w] for w in CURRICULUM]))):.3f}")

print("\n=== ПОСТРОЕНИЕ ГРАФИКА ===\n")
stages = [r[0] for r in results]
accuracies = [r[1] for r in results]

plt.figure(figsize=(10, 6))
plt.plot(stages, accuracies, 'o-', linewidth=2, markersize=8, color='#2ca02c')
plt.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Цель (90%)')
plt.xlabel('Количество слов в обучении', fontsize=12)
plt.ylabel('Точность на всех известных словах (%)', fontsize=12)
plt.title('Цифровое детство: постепенное обучение ConceptNet', fontsize=14, fontweight='bold')
plt.ylim(0, 105)
plt.grid(True, alpha=0.3)
plt.legend()

for x, y in zip(stages, accuracies):
    plt.annotate(f'{y:.0f}%', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

plt.tight_layout()
plt.savefig('curriculum_learning_plot.png', dpi=150)
print("✅ График сохранен в curriculum_learning_plot.png")

torch.save(model.state_dict(), 'concept_net_curriculum.pth')
print("✅ Модель сохранена в concept_net_curriculum.pth")
