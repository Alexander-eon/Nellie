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

# Загружаем данные
with open('extended_words.pkl', 'rb') as f:
    data = pickle.load(f)

words = data['words']
embeddings = data['embeddings']

print(f"=== ОБУЧЕНИЕ НА {len(words)} СЛОВАХ (ДВУХВЕТОЧНАЯ СЕТЬ) ===\n")

# === ПРОЦЕНТИЛЬНАЯ КЛАССИФИКАЦИЯ ===
def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2) + 1e-8)

danger_key = embeddings.get('опасность')
food_key = embeddings.get('еда')
calm_key = embeddings.get('спокойствие')

sim_danger = {word: cosine_sim(emb, danger_key) for word, emb in embeddings.items()}
sim_food = {word: cosine_sim(emb, food_key) for word, emb in embeddings.items()}
sim_calm = {word: cosine_sim(emb, calm_key) for word, emb in embeddings.items()}

sorted_danger = sorted(sim_danger.keys(), key=lambda x: sim_danger[x], reverse=True)
sorted_food = sorted(sim_food.keys(), key=lambda x: sim_food[x], reverse=True)
sorted_calm = sorted(sim_calm.keys(), key=lambda x: sim_calm[x], reverse=True)

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

print("=== КЛАССИФИКАЦИЯ ===")
print(f"  Опасные: {sum(1 for w in words if word_class[w]=='danger')}")
print(f"  Еда: {sum(1 for w in words if word_class[w]=='food')}")
print(f"  Спокойные: {sum(1 for w in words if word_class[w]=='calm')}")
print(f"  Нейтральные: {sum(1 for w in words if word_class[w]=='neutral')}\n")

# === PCA ===
X = np.array([embeddings[w] for w in words])
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)
X_min, X_max = X_reduced.min(axis=0), X_reduced.max(axis=0)
X_normalized = (X_reduced - X_min) / (X_max - X_min + 1e-8)

REDUCED = {word: X_normalized[i] for i, word in enumerate(words)}

# === ДВУХВЕТОЧНАЯ НЕЙРОСЕТЬ ===
class DualContextNet(nn.Module):
    def __init__(self, embed_size=50, context_size=3, hidden=16):
        super().__init__()
        # Ветка для слова
        self.fc_embed = nn.Linear(embed_size, hidden)
        self.tanh_embed = nn.Tanh()
        # Ветка для контекста
        self.fc_context = nn.Linear(context_size, hidden)
        self.tanh_context = nn.Tanh()
        # Слияние
        self.fc_merge = nn.Linear(hidden * 2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embed = x[:, :50]
        context = x[:, 50:]
        
        h_embed = self.tanh_embed(self.fc_embed(embed))
        h_context = self.tanh_context(self.fc_context(context))
        
        merged = torch.cat([h_embed, h_context], dim=1)
        out = self.sigmoid(self.fc_merge(merged))
        return out

model = DualContextNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# === ГЕНЕРАЦИЯ ДАННЫХ (ИСПРАВЛЕННЫЕ ФОРМУЛЫ) ===
print("=== ГЕНЕРАЦИЯ ДАННЫХ (10000 ПРИМЕРОВ) ===\n")
X_train = []
y_train = []

def get_target_action(word, pain, hunger, fear):
    cls = word_class[word]
    
    if cls == 'danger':
        # Испуганный -> 1.0, спокойный -> 0.1
        return 0.1 + (pain + fear) * 0.45
    elif cls == 'food':
        # Голодный -> 0.9, сытый -> 0.1 (ИСПРАВЛЕНО!)
        return 0.1 + hunger * 0.8
    elif cls == 'calm':
        return 0.5
    else:
        return 0.5

for _ in range(10000):
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
print("=== ОБУЧЕНИЕ (300 ЭПОХ) ===\n")
for epoch in range(300):
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"  Эпоха {epoch}: Loss = {loss.item():.4f}")

# === ФИНАЛЬНЫЙ ТЕСТ ===
print("\n=== ФИНАЛЬНЫЙ ТЕСТ (ПРОВЕРКА ЗАЗЕМЛЕНИЯ) ===\n")

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

torch.save(model.state_dict(), 'dual_context_net.pth')
print("✅ Модель сохранена в dual_context_net.pth")
