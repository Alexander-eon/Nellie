"""
Анализ сообщений: есть ли связь между target_gpu/target_cpu и message?

Если message_a устойчиво связан с target_gpu — A кодирует цель для B.
Если message_b устойчиво связан с target_cpu — B кодирует цель для A.
"""

import sys
import glob
import os
import numpy as np

sys.path.insert(0, '.')


def analyze_pair(path: str):
    try:
        data = np.load(path)
    except Exception as e:
        return None

    # Проверяем, есть ли messages
    if 'messages_a' not in data or 'messages_b' not in data:
        return None

    messages_a = data['messages_a'].flatten()
    messages_b = data['messages_b'].flatten()
    actions_a = data['actions_a'].flatten()
    actions_b = data['actions_b'].flatten()
    target_cpu = data['target_cpu']
    target_gpu = data['target_gpu']
    cpu = data['cpu']
    gpu = data['gpu']

    n = len(messages_a)
    if n < 100:
        return None

    # Корреляции: message_a vs target_gpu, message_b vs target_cpu
    corr_ma_tg = np.corrcoef(messages_a[:-1], target_gpu[1:])[0, 1]
    corr_mb_tc = np.corrcoef(messages_b[:-1], target_cpu[1:])[0, 1]

    # Корреляция message_a с CPU (влияет ли message на что-то)
    corr_ma_cpu = np.corrcoef(messages_a[:-1], cpu[1:])[0, 1]
    corr_mb_gpu = np.corrcoef(messages_b[:-1], gpu[1:])[0, 1]

    return {
        "name": os.path.basename(path).replace('.npz', ''),
        "steps": n,
        "corr_ma_tg": corr_ma_tg,
        "corr_mb_tc": corr_mb_tc,
        "corr_ma_cpu": corr_ma_cpu,
        "corr_mb_gpu": corr_mb_gpu,
        "std_ma": messages_a.std(),
        "std_mb": messages_b.std(),
    }


if __name__ == "__main__":
    files = sorted(glob.glob('/tmp/sep_evo_gen2_pair*.npz'))
    print(f'Файлов: {len(files)}')
    print()

    results = []
    for f in files:
        r = analyze_pair(f)
        if r:
            results.append(r)

    if not results:
        print("Нет данных с messages. Проверь, что прогон был после правок.")
        sys.exit(0)

    # Сортируем по силе связи message_a с target_gpu
    results.sort(key=lambda x: abs(x["corr_ma_tg"]), reverse=True)

    print(f'{"пара":>20} | {"шагов":>6} | {"std(ma)":>7} | {"corr(ma,tg)":>12} | {"std(mb)":>7} | {"corr(mb,tc)":>12}')
    print("-" * 90)
    for r in results:
        print(f'{r["name"]:>20} | {r["steps"]:>6} | {r["std_ma"]:>7.3f} | '
              f'{r["corr_ma_tg"]:>+12.3f} | {r["std_mb"]:>7.3f} | {r["corr_mb_tc"]:>+12.3f}')

    print()
    print("Интерпретация:")
    print("  corr(ma, tg) — связь message_a с target_gpu (A кодирует цель для B)")
    print("  corr(mb, tc) — связь message_b с target_cpu (B кодирует цель для A)")
    print()
    print(f'Максимальная |corr(ma, tg)|: {max(abs(r["corr_ma_tg"]) for r in results):.3f}')
    print(f'Максимальная |corr(mb, tc)|: {max(abs(r["corr_mb_tc"]) for r in results):.3f}')
