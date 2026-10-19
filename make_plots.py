import matplotlib.pyplot as plt
import numpy as np

# --- Данные для Графика 1 (Эволюция) ---
generations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
correlation_evolution = [0.945, 0.992, 0.994, 0.990, 0.997, 0.993, 0.994, 0.986, 0.997, 0.995]

# --- Данные для Графика 2 (Острова) ---
gens_islands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
island_1_fast = [0.952, 0.994, 0.996, 0.997, 0.998, 0.998, 0.998, 0.999, 0.999, 0.998]
island_2_slow = [0.170, 0.899, 0.894, 0.939, 0.936, 0.932, 0.885, 0.918, 0.894, 0.883]

# Настройка стиля (научный вид)
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# === ГРАФИК 1 ===
ax1.plot(generations, correlation_evolution, 'o-', color='#2ca02c', linewidth=2, markersize=8, label='Эволюция языка')
ax1.axhline(y=0.997, color='red', linestyle='--', alpha=0.7, label='Предел Шеннона (0.997)')
ax1.set_title('Эволюция языка: достижение предела канала', fontsize=14, fontweight='bold')
ax1.set_xlabel('Поколение', fontsize=12)
ax1.set_ylabel('Корреляция (msg_A vs target_cpu)', fontsize=12)
ax1.set_ylim(0.9, 1.01)
ax1.legend()
ax1.grid(True, alpha=0.3)

# === ГРАФИК 2 ===
ax2.plot(gens_islands, island_1_fast, 's-', color='#1f77b4', linewidth=2, markersize=8, label='Остров 1 (Быстрый)')
ax2.plot(gens_islands, island_2_slow, '^-', color='#ff7f0e', linewidth=2, markersize=8, label='Остров 2 (Медленный)')
ax2.axhline(y=0.973, color='#1f77b4', linestyle=':', alpha=0.7, label='Гибрид в быстрой среде (0.973)')
ax2.axhline(y=0.569, color='#ff7f0e', linestyle=':', alpha=0.7, label='Гибрид в медленной среде (0.569)')
ax2.set_title('Островная модель: рождение диалектов', fontsize=14, fontweight='bold')
ax2.set_xlabel('Поколение', fontsize=12)
ax2.set_ylabel('Корреляция', fontsize=12)
ax2.set_ylim(0.0, 1.05)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('language_evolution_plots.png', dpi=300)
print("Графики сохранены в файл: language_evolution_plots.png")
