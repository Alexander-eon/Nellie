"""
Пластичная нейронная матрица.
Однородный граф нейронов. Никакого backprop. Только Хебб + награда.

Архитектура:
- N нейронов, все связаны со всеми (N×N весов).
- Первые N_IN нейронов — входные.
- Последние N_OUT нейронов — выходные.
- Активации обновляются каждый такт.

Пластичность:
    Δw_ij = α · R · (x_i · y_j) + β · ξ

где:
- α — скорость обучения,
- R — награда (−1..+1),
- x_i · y_j — произведение активностей (правило Хебба),
- β — сила мутации,
- ξ — случайный шум.
"""

import numpy as np


class Brain:
    def __init__(
        self,
        n_neurons: int = 32,
        n_input: int = 14,
        n_output: int = 2,
        alpha: float = 0.001,
        beta: float = 0.0001,
        trace_lambda: float = 0.95,
        seed: int = 42,
    ):
        self.n = n_neurons
        self.n_in = n_input
        self.n_out = n_output
        self.alpha = alpha
        self.beta = beta
        self.trace_lambda = trace_lambda

        self.rng = np.random.default_rng(seed)

        # Веса: N×N, малые случайные
        self.weights = self.rng.normal(0.0, 0.1, (self.n, self.n))

        # Активации: N
        self.activations = np.zeros(self.n)
        self.prev_activations = np.zeros(self.n)

        # Eligibility trace: N×N (накопление недавней активности)
        self.trace = np.zeros((self.n, self.n))

        # Скользящее среднее награды (baseline)
        self.reward_baseline = 0.0
        self.baseline_alpha = 0.01

    def forward(self, input_vec: np.ndarray) -> np.ndarray:
        """
        Один такт.
        input_vec: вектор длины n_in.
        Возвращает: вектор длины n_out (действия).
        """
        # Сохраняем предыдущие активации
        self.prev_activations = self.activations.copy()

        # Подаём вход в первые n_in нейронов
        self.activations[: self.n_in] = input_vec

        # Обновляем остальные нейроны
        # (последние n_out — это выход, они тоже обновляются)
        # Считаем сумму входов
        new_acts = np.tanh(self.weights.T @ self.activations)

        # Первые n_in нейронов остаются входными
        new_acts[: self.n_in] = input_vec

        # Записываем новые активации
        self.activations = new_acts

        # Выход: последние n_out нейронов
        out = self.activations[-self.n_out:]

        # Приводим выход к [0, 1] через сигмоиду
        out = 1.0 / (1.0 + np.exp(-out))

        return out

    def update_weights(self, reward: float):
        """
        Обновление весов по правилу Хебба с модуляцией наградой.
        """
        # Обновляем baseline
        self.reward_baseline = (
            (1 - self.baseline_alpha) * self.reward_baseline
            + self.baseline_alpha * reward
        )

        # Центрируем награду
        r_centered = reward - self.reward_baseline

        # Хеббовское произведение активностей
        hebb = np.outer(self.activations, self.activations)

        # Обновляем eligibility trace
        self.trace = (
            self.trace_lambda * self.trace
            + (1 - self.trace_lambda) * hebb
        )

        # Основное обновление: α · R · trace
        delta = self.alpha * r_centered * self.trace

        # Мутация: малый шум
        noise = self.rng.normal(0.0, self.beta, (self.n, self.n))

        # Итоговое обновление
        self.weights += delta + noise

        # Клиппинг: не даём весам улететь
        np.clip(self.weights, -3.0, 3.0, out=self.weights)

    def get_state(self) -> dict:
        """Состояние мозга для сохранения."""
        return {
            "weights": self.weights.copy(),
            "activations": self.activations.copy(),
            "trace": self.trace.copy(),
            "reward_baseline": self.reward_baseline,
        }

    def load_state(self, state: dict):
        """Загрузка состояния."""
        self.weights = state["weights"].copy()
        self.activations = state["activations"].copy()
        self.trace = state["trace"].copy()
        self.reward_baseline = state["reward_baseline"]
