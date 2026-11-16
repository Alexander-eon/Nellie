import numpy as np
import os
import pickle
import time
import ollama
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from numpy.linalg import norm

# === ЗАГРУЗКА ДАННЫХ ===
with open('extended_words.pkl', 'rb') as f:
    data = pickle.load(f)

words = data['words']
embeddings = data['embeddings']

print(f"=== ОБУЧЕНИЕ С УСИЛЕННЫМ СЛОВОМ ({len(words)} слов) ===\n")

# === КЛАССИФИКАЦИЯ (как раньше) ===
def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2) + 1e-8)

danger_key = embeddings.get('опасность')
food_key = embeddings.get('еда')
calm_key = embeddings.get('спокойствие')

sim_danger = {w: cosine_sim(embeddings[w], danger_key) for w in words}
sim_food = {w: cosine_sim(embeddings[w], food_key) for w in words}
sim_calm = {w: cosine_sim(embeddings[w], calm_key) for w in words}

sorted_danger = sorted(words, key=lambda x: sim_danger[x], reverse=True)
sorted_food = sorted(words, key=lambda x: sim_food[x], reverse=True)
sorted_calm = sorted(words, key=lambda x: sim_calm[x], reverse=True)

n = len(words)
top_danger = set(sorted_danger[:n//5])
top_food = set(sorted_food[:n//5])
top_calm = set(sorted_calm[:n//5])

word_class = {}
for word in words:
    if word in top_danger: word_class[word] = 'danger'
    elif word in top_food: word_class[word] = 'food'
    elif word in top_calm: word_class[word] = 'calm'
    else: word_class[word] = 'neutral'

print(f"Опасные: {sum(1 for w in words if word_class[w]=='danger')}")
print(f"Еда: {sum(1 for w in words if word_class[w]=='food')}")
print(f"Спокойные: {sum(1 for w in words if word_class[w]=='calm')}")
print(f"Нейтральные: {sum(1 for w in words if word_class[w]=='neutral')}\n")

# === PCA ===
X = np.array([embeddings[w] for w in words])
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)
X_min, X_max = X_reduced.min(axis=0), X_reduced.max(axis=0)
X_normalized = (X_reduced - X_min) / (X_max - X_min + 1e-8)
REDUCED = {word: X_normalized[i] for i, word in enumerate(words)}

# === НОВАЯ ФУНКЦИЯ ЦЕЛИ: СЛОВО ГЛАВНЕЕ ===
def get_target_action(word, pain, hunger, fear):
    cls = word_class[word]
    
    if cls == 'danger':
        # Опасность: базовое 0.9, контекст корректирует на ±0.15
        base = 0.9
        context_boost = (pain + fear) * 0.15 - 0.1  # от -0.1 до +0.2
        return np.clip(base + context_boost, 0.0, 1.0)
    elif cls == 'food':
        # Еда: базовое 0.2, контекст корректирует на ±0.15
        base = 0.2
        context_boost = hunger * 0.3 - 0.1  # от -0.1 до +0.2
        return np.clip(base + context_boost, 0.0, 1.0)
    elif cls == 'calm':
        # Спокойствие: базовое 0.1, контекст почти не влияет
        return 0.1 + (pain + fear) * 0.05
    else:
        # Нейтральные: 0.5, контекст слабо
        return 0.5 + (pain + fear) * 0.05

# === ДВУХВЕТОЧНАЯ СЕТЬ (усиленная) ===
class StrongDualNet(nn.Module):
    def __init__(self, embed_size=50, context_size=3, hidden=16):
        super().__init__()
        # Ветка слова (более мощная)
        self.fc_embed = nn.Linear(embed_size, hidden * 2)
        self.tanh_embed = nn.Tanh()
        # Ветка контекста (слабее)
        self.fc_context = nn.Linear(context_size, hidden // 2)
        self.tanh_context = nn.Tanh()
        # Слияние
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
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# === ГЕНЕРАЦИЯ ДАННЫХ ===
print("=== ГЕНЕРАЦИЯ ДАННЫХ (20000 ПРИМЕРОВ) ===\n")
X_train = []
y_train = []

for _ in range(20000):
    word = np.random.choice(words)
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

# === ОБУЧЕНИЕ ===
print("=== ОБУЧЕНИЕ (500 ЭПОХ) ===\n")
for epoch in range(500):
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"  Эпоха {epoch}: Loss = {loss.item():.4f}")

# === ФИНАЛЬНЫЙ ТЕСТ ===
print("\n=== ФИНАЛЬНЫЙ ТЕСТ (СИЛЬНАЯ ДИФФЕРЕНЦИАЦИЯ) ===\n")

contexts = {
    'сытый спокойный': [0.0, 0.0, 0.0],
    'голодный': [0.0, 1.0, 0.0],
    'испуганный': [1.0, 0.0, 1.0],
}

test_words = ['еда', 'опасность', 'спокойствие', 'собака', 'огонь']

for word in test_words:
    if word not in REDUCED:
        continue
    emb = REDUCED[word]
    print(f"Слово: '{word}' (класс: {word_class[word]})")
    for name, ctx in contexts.items():
        inp = np.concatenate([emb, np.array(ctx)])
        with torch.no_grad():
            pred = model(torch.tensor(inp, dtype=torch.float32).unsqueeze(0)).item()
        print(f"  {name:20}: {pred:.3f}")
    print()

# Проверка: разница между "едой" и "опасностью"
food_emb = REDUCED['еда']
danger_emb = REDUCED['опасность']
ctx = np.array([0.5, 0.8, 0.5])  # Голодный и испуганный

inp_food = np.concatenate([food_emb, ctx])
inp_danger = np.concatenate([danger_emb, ctx])

with torch.no_grad():
    pred_food = model(torch.tensor(inp_food, dtype=torch.float32).unsqueeze(0)).item()
    pred_danger = model(torch.tensor(inp_danger, dtype=torch.float32).unsqueeze(0)).item()

print(f"Разница между 'едой' и 'опасностью': {abs(pred_food - pred_danger):.3f}")
if abs(pred_food - pred_danger) > 0.3:
    print("✅ УСПЕХ! Слова сильно дифференцированы.")
else:
    print("⚠️ Разница всё ещё мала. Нужна доработка.")

torch.save(model.state_dict(), 'strong_dual_net.pth')
print("\n✅ Модель сохранена в strong_dual_net.pth")
