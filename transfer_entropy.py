"""
Transfer entropy: направленная передача информации.

TE(A→B) = I(B_future ; A_past | B_past)

Измеряет: насколько знание A_past улучшает предсказание B_future,
СВЕРХ того, что уже даёт B_past.

Если TE > 0 и значим → A влияет на B.
Если TE ≈ 0 → A и B независимы.
"""

import numpy as np


def discretize(x: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Дискретизирует непрерывный сигнал в n_bins корзин."""
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.flatten()
    # Убираем NaN
    x = np.nan_to_num(x, nan=0.0)
    # Равномерные интервалы
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-10:
        return np.zeros_like(x, dtype=int)
    bins = np.linspace(x_min, x_max + 1e-10, n_bins + 1)
    return np.digitize(x, bins[:-1]) - 1


def entropy(counts: np.ndarray) -> float:
    """Энтропия по частотам."""
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def joint_entropy(*arrays: np.ndarray) -> float:
    """Совместная энтропия нескольких дискретных массивов."""
    stacked = np.stack(arrays, axis=1)
    # Уникальные комбинации
    unique, counts = np.unique(stacked, axis=0, return_counts=True)
    return entropy(counts)


def transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    history: int = 1,
    n_bins: int = 5,
) -> float:
    """
    TE(source → target).

    source: сигнал-источник (1D)
    target: сигнал-цель (1D)
    history: сколько прошлых значений учитывать
    n_bins: число корзин для дискретизации

    Формула:
    TE = H(target_future | target_past) - H(target_future | target_past, source_past)
    """
    src = discretize(source, n_bins)
    tgt = discretize(target, n_bins)

    n = len(src)
    if n < history + 2:
        return 0.0

    # Формируем прошлое и будущее
    tgt_future = tgt[history:]
    tgt_past_list = [tgt[history - 1 - i : n - 1 - i] for i in range(history)]
    src_past_list = [src[history - 1 - i : n - 1 - i] for i in range(history)]

    # H(target_future | target_past) = H(target_future, target_past) - H(target_past)
    h_tgt_future_and_past = joint_entropy(tgt_future, *tgt_past_list)
    h_tgt_past = joint_entropy(*tgt_past_list)

    # H(target_future | target_past, source_past)
    h_tgt_future_and_all = joint_entropy(tgt_future, *tgt_past_list, *src_past_list)
    h_all_past = joint_entropy(*tgt_past_list, *src_past_list)

    # TE = [H(f,p) - H(p)] - [H(f,p,s) - H(p,s)]
    cond_1 = h_tgt_future_and_past - h_tgt_past
    cond_2 = h_tgt_future_and_all - h_all_past

    te = cond_1 - cond_2
    return max(0.0, float(te))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from comm_agent import CommAgent

    print("=== Тест transfer entropy ===")
    print()

    # === Тест 1: независимые сигналы ===
    print("Тест 1: независимые сигналы (должно быть ~0)")
    rng = np.random.default_rng(42)
    x1 = rng.normal(0, 1, 1000)
    y1 = rng.normal(0, 1, 1000)
    te = transfer_entropy(x1, y1, history=1, n_bins=5)
    print(f"  TE(x → y) = {te:.6f}")
    print()

    # === Тест 2: y зависит от x со сдвигом ===
    print("Тест 2: y = сдвинутый x (должно быть > 0)")
    x2 = rng.normal(0, 1, 1000)
    y2 = np.roll(x2, 1) + rng.normal(0, 0.1, 1000)
    y2[0] = 0
    te = transfer_entropy(x2, y2, history=1, n_bins=5)
    print(f"  TE(x → y) = {te:.6f}")
    print()

    # === Тест 3: на реальных агентах ===
    print("Тест 3: реальные агенты (2000 тактов)")
    agent = CommAgent(dt=0.05)
    agent.start()

    actions_a = []
    actions_b = []
    sensors_log = []

    try:
        for i in range(2000):
            log = agent.step()
            if log['step'] > 0:
                actions_a.append(log['a0'])
                actions_b.append(log['b0'])
                _, state = agent.sensor.read()
                sensors_log.append(state['cpu_load'])
    finally:
        agent.stop()

    actions_a = np.array(actions_a)
    actions_b = np.array(actions_b)
    sensors_log = np.array(sensors_log)

    print(f"  Тактов: {len(actions_a)}")
    print()

    te_a_to_b = transfer_entropy(actions_a, actions_b, history=1, n_bins=5)
    te_b_to_a = transfer_entropy(actions_b, actions_a, history=1, n_bins=5)
    te_a_to_s = transfer_entropy(actions_a, sensors_log, history=1, n_bins=5)
    te_b_to_s = transfer_entropy(actions_b, sensors_log, history=1, n_bins=5)

    print("  === Transfer Entropy ===")
    print(f"  A → B  : {te_a_to_b:.6f}")
    print(f"  B → A  : {te_b_to_a:.6f}")
    print(f"  A → S  : {te_a_to_s:.6f}")
    print(f"  B → S  : {te_b_to_s:.6f}")
    print()
    print("  Интерпретация:")
    print(f"  - TE(A→B) > 0 : A влияет на B")
    print(f"  - TE(B→A) > 0 : B влияет на A")
    print(f"  - TE(A→S) > 0 : A влияет на сенсоры (тело)")
    print(f"  - TE(B→S) > 0 : B влияет на сенсоры (тело)")
