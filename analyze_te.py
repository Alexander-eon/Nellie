"""
Анализ TE по всем парам поколения.

Загружает файлы /tmp/sep_evo_gen{N}_pair*.npz,
считает TE(target_gpu → cpu) и TE(target_cpu → gpu),
выводит топ и средние.
"""

import sys
import glob
import os
import numpy as np

sys.path.insert(0, '.')
from transfer_entropy import transfer_entropy


def analyze_generation(gen: int, min_steps: int = 200):
    """Анализирует одну поколение."""
    pattern = f"/tmp/sep_evo_gen{gen}_pair*.npz"
    files = sorted(glob.glob(pattern))

    if not files:
        return None

    results = []
    for f in files:
        try:
            data = np.load(f)
            cpu = data['cpu']
            if len(cpu) < min_steps:
                continue

            actions_a = data['actions_a'].flatten()
            actions_b = data['actions_b'].flatten()
            gpu = data['gpu']
            target_cpu = data['target_cpu']
            target_gpu = data['target_gpu']

            te_tg_cpu = transfer_entropy(target_gpu, cpu, history=1, n_bins=5)
            te_tc_gpu = transfer_entropy(target_cpu, gpu, history=1, n_bins=5)

            name = os.path.basename(f).replace(f'sep_evo_gen{gen}_', '').replace('.npz', '')
            results.append({
                "name": name,
                "te_tg_cpu": te_tg_cpu,
                "te_tc_gpu": te_tc_gpu,
                "steps": len(cpu),
            })
        except Exception as e:
            print(f"  Ошибка в {os.path.basename(f)}: {e}")

    return results


def print_generation(gen: int, results: list):
    """Красиво выводит результаты поколения."""
    if not results:
        print(f"Поколение {gen}: нет данных")
        return

    # Сортируем по TE(tc → GPU)
    results_sorted = sorted(results, key=lambda x: x["te_tc_gpu"], reverse=True)

    print(f"=== Поколение {gen} ({len(results)} пар) ===")
    print()
    print("Топ-5 по TE(tc → GPU):")
    print(f"  {'пара':>8} | {'шагов':>6} | {'tg→CPU':>8} | {'tc→GPU':>8}")
    print("-" * 42)
    for r in results_sorted[:5]:
        print(f"  {r['name']:>8} | {r['steps']:>6} | {r['te_tg_cpu']:>8.4f} | {r['te_tc_gpu']:>8.4f}")

    avg_tg = np.mean([r["te_tg_cpu"] for r in results])
    avg_tc = np.mean([r["te_tc_gpu"] for r in results])
    max_tc = max([r["te_tc_gpu"] for r in results])

    print()
    print("Средние:")
    print(f"  tg→CPU: {avg_tg:.4f}")
    print(f"  tc→GPU: {avg_tc:.4f}")
    print(f"  tc→GPU (лучшая): {max_tc:.4f}")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", type=int, default=9, help="Поколение")
    parser.add_argument("--all", action="store_true", help="Анализ всех поколений")
    args = parser.parse_args()

    if args.all:
        for gen in range(10):
            results = analyze_generation(gen)
            if results:
                print_generation(gen, results)
    else:
        results = analyze_generation(args.gen)
        if results:
            print_generation(args.gen, results)
        else:
            print(f"Поколение {args.gen}: нет данных")


if __name__ == "__main__":
    main()
