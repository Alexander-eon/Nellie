"""
Мутации и наследование для эволюции.
"""

import numpy as np


def mutate_weights(weights: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    Добавляет гауссов шум к весам.
    sigma — сила мутации.
    """
    noise = rng.normal(0.0, sigma, weights.shape)
    mutated = weights + noise
    # Клиппинг
    return np.clip(mutated, -3.0, 3.0)


def crossover(w1: np.ndarray, w2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Смешивает веса двух родителей: случайно берём от каждого.
    """
    mask = rng.random(w1.shape) < 0.5
    child = np.where(mask, w1, w2)
    return child


def clone_brain(parent_brain, sigma: float = 0.01, seed: int = None) -> object:
    """
    Создаёт нового агента с весами, мутированными от родителя.
    Возвращает объект, который можно передать в Agent.
    """
    from brain import Brain

    rng = np.random.default_rng(seed)

    # Создаём новый мозг
    child_brain = Brain(seed=seed)
    # Копируем веса родителя с мутацией
    child_brain.weights = mutate_weights(parent_brain.weights, sigma, rng)
    # Активации и trace не наследуются — начинаем с нуля
    child_brain.activations = np.zeros_like(parent_brain.activations)
    child_brain.trace = np.zeros_like(parent_brain.trace)
    child_brain.reward_baseline = parent_brain.reward_baseline

    return child_brain
