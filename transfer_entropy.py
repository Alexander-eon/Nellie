"""
Transfer entropy: направленная передача информации.

TE(A→B) = I(B_future ; A_past | B_past)

Измеряет: насколько знание A_past улучшает предсказание B_future,
СВЕРХ того, что уже даёт B_past.
"""

import numpy as np


def discretize(x: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Дискретизирует сигнал в n_bins равномерных корзин."""
    x = np.asarray(x, dtype=float).flatten()
    x = np.nan_to_num(x, nan=0.0)
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-10:
        return np.zeros_like(x, dtype=int)
    bins = np.linspace(x_min, x_max + 1e-10, n_bins + 1)
    return np.digitize(x, bins[:-1]) - 1


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _joint_entropy_pairs(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Совместная энтропия двух 1D массивов одинаковой длины."""
    if len(arr1) != len(arr2):
        n = min(len(arr1), len(arr2))
        arr1 = arr1[:n]
        arr2 = arr2[:n]
    stacked = np.stack([arr1, arr2], axis=1)
    _, counts = np.unique(stacked, axis=0, return_counts=True)
    return _entropy_from_counts(counts)


def _joint_entropy_triple(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Совместная энтропия трёх 1D массивов одинаковой длины."""
    n = min(len(a), len(b), len(c))
    a, b, c = a[:n], b[:n], c[:n]
    stacked = np.stack([a, b, c], axis=1)
    _, counts = np.unique(stacked, axis=0, return_counts=True)
    return _entropy_from_counts(counts)


def transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    history: int = 1,
    n_bins: int = 5,
) -> float:
    """
    TE(source -> target) при history=1.

    TE = H(t_f | t_p) - H(t_f | t_p, s_p)
       = [H(t_f, t_p) - H(t_p)] - [H(t_f, t_p, s_p) - H(t_p, s_p)]
    """
    src = discretize(source, n_bins)
    tgt = discretize(target, n_bins)

    n = min(len(src), len(tgt))
    if n < history + 10:
        return 0.0

    src = src[:n]
    tgt = tgt[:n]

    # Формируем массивы
    tgt_future = tgt[history:]                # длина n-history
    tgt_past = tgt[:n - history]              # длина n-history
    src_past = src[:n - history]              # длина n-history

    if len(tgt_future) != len(tgt_past) or len(tgt_future) != len(src_past):
        return 0.0

    # H(t_f, t_p) - H(t_p)
    h_tf_tp = _joint_entropy_pairs(tgt_future, tgt_past)
    h_tp = _entropy_from_counts(np.bincount(tgt_past))

    # H(t_f, t_p, s_p) - H(t_p, s_p)
    h_tf_tp_sp = _joint_entropy_triple(tgt_future, tgt_past, src_past)
    h_tp_sp = _joint_entropy_pairs(tgt_past, src_past)

    te = (h_tf_tp - h_tp) - (h_tf_tp_sp - h_tp_sp)
    return max(0.0, float(te))


if __name__ == "__main__":
    # Простой тест
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 2000)
    y = np.roll(x, 1) + rng.normal(0, 0.1, 2000)
    y[0] = 0
    z = rng.normal(0, 1, 2000)

    print(f"TE(x -> y): {transfer_entropy(x, y):.4f}  (должно быть > 0)")
    print(f"TE(x -> z): {transfer_entropy(x, z):.4f}  (должно быть ~0)")
