"""
Анализ словаря: какие действия A и B выбирают при разных целях.

Для пары:
1. Дискретизируем target_gpu (округляем до уровня).
2. Смотрим распределение действий A при каждом уровне.
3. Если распределение устойчиво (одно действие доминирует) — есть слово.
"""

import sys
import numpy as np

sys.path.insert(0, '.')


def analyze_pair(path: str, name: str):
    data = np.load(path)
    actions_a = data['actions_a'].flatten()
    actions_b = data['actions_b'].flatten()
    target_cpu = data['target_cpu']
    target_gpu = data['target_gpu']
    cpu = data['cpu']
    gpu = data['gpu']

    n = len(actions_a)
    if n < 100:
        print(f'{name}: слишком мало тактов ({n})')
        return

    print(f'=== {name} ({n} тактов) ===')
    print()

    # Дискретизация цели target_gpu в 5 корзин
    bins = np.linspace(target_gpu.min(), target_gpu.max() + 1e-9, 6)

    print('A: target_gpu -> какие действия?')
    print(f'  {"target_gpu":>12} | {"n":>5} | {"action_a (mean)":>15} | {"std":>6}')
    print('-' * 55)
    for i in range(5):
        mask = (target_gpu >= bins[i]) & (target_gpu < bins[i + 1])
        if mask.sum() < 20:
            continue
        acts = actions_a[mask]
        print(f'  {bins[i]:.2f}-{bins[i+1]:.2f} | {mask.sum():>5} | '
              f'{acts.mean():>15.3f} | {acts.std():>6.3f}')

    print()
    print('B: target_cpu -> какие действия?')
    bins2 = np.linspace(target_cpu.min(), target_cpu.max() + 1e-9, 6)
    print(f'  {"target_cpu":>12} | {"n":>5} | {"action_b (mean)":>15} | {"std":>6}')
    print('-' * 55)
    for i in range(5):
        mask = (target_cpu >= bins2[i]) & (target_cpu < bins2[i + 1])
        if mask.sum() < 20:
            continue
        acts = actions_b[mask]
        print(f'  {bins2[i]:.2f}-{bins2[i+1]:.2f} | {mask.sum():>5} | '
              f'{acts.mean():>15.3f} | {acts.std():>6.3f}')
    print()


if __name__ == "__main__":
    import glob
    files = sorted(glob.glob('/tmp/sep_evo_gen9_pair*.npz'))
    print(f'Найдено файлов: {len(files)}')
    print()

    # Анализ всех пар, но вывод — топ-3 по TE
    for f in files:
        name = f.split('/')[-1].replace('.npz', '')
        if 'pair2' in name or 'pair8' in name or 'pair1' in name:
            analyze_pair(f, name)
