"""
Модель мира: предсказывает следующее состояние тела.

Вход:  [S(t), A(t)] — 12 сенсоров + 2 действия = 14 чисел
Выход: S_pred(t+1) — 12 сенсоров

Обучение: онлайн-градиент (delta rule), без backprop через тело.
Тело — чёрный ящик. Модель просто учится предсказывать.
"""

import numpy as np


class WorldModel:
    def __init__(
        self,
        n_sensors: int = 12,
        n_actions: int = 2,
        learning_rate: float = 0.01,
        forget_factor: float = 0.999,
        seed: int = 43,
    ):
        self.n_s = n_sensors
        self.n_a = n_actions
        self.n_in = n_sensors + n_actions
        self.lr = learning_rate
        self.forget = forget_factor

        self.rng = np.random.default_rng(seed)

        # Веса: (n_sensors, n_in)
        self.W = self.rng.normal(0.0, 0.01, (self.n_s, self.n_in))
        self.b = np.zeros(self.n_s)

        # Статистика для нормализации сенсоров
        self.sensor_mean = np.zeros(self.n_s)
        self.sensor_std = np.ones(self.n_s)
        self.stats_alpha = 0.01

        # Предыдущий вход (для обновления)
        self.last_input = None
        self.last_prediction = None

    def _normalize(self, sensors: np.ndarray) -> np.ndarray:
        """Нормализует сенсоры по скользящей статистике."""
        return (sensors - self.sensor_mean) / (self.sensor_std + 1e-6)

    def _update_stats(self, sensors: np.ndarray):
        """Обновляет статистику по новому наблюдению."""
        self.sensor_mean = (
            (1 - self.stats_alpha) * self.sensor_mean
            + self.stats_alpha * sensors
        )
        self.sensor_std = (
            (1 - self.stats_alpha) * self.sensor_std
            + self.stats_alpha * np.abs(sensors - self.sensor_mean)
        )

    def predict(self, sensors: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Предсказывает S(t+1).
        sensors: 12 чисел
        actions: 2 числа
        Возвращает: 12 чисел (предсказанные сенсоры)
        """
        s_norm = self._normalize(sensors)
        x = np.concatenate([s_norm, actions])

        pred_norm = self.W @ x + self.b
        # Денормализация
        pred = pred_norm * (self.sensor_std + 1e-6) + self.sensor_mean

        self.last_input = x
        self.last_prediction = pred_norm

        return pred

    def update(self, actual_next: np.ndarray) -> float:
        """
        Обновляет модель по фактическому S(t+1).
        Возвращает ошибку предсказания (до обновления).
        """
        if self.last_input is None:
            return 0.0

        # Нормализуем фактическое значение
        actual_norm = (
            (actual_next - self.sensor_mean) / (self.sensor_std + 1e-6)
        )

        # Ошибка
        error = actual_norm - self.last_prediction

        # Градиентный шаг
        self.W += self.lr * np.outer(error, self.last_input)
        self.b += self.lr * error

        # Обновляем статистику
        self._update_stats(actual_next)

        # Возвращаем ошибку в оригинальном масштабе
        pred_orig = (
            self.last_prediction * (self.sensor_std + 1e-6) + self.sensor_mean
        )
        error_orig = np.abs(actual_next - pred_orig).mean()

        return float(error_orig)

    def get_state(self) -> dict:
        return {
            "W": self.W.copy(),
            "b": self.b.copy(),
            "sensor_mean": self.sensor_mean.copy(),
            "sensor_std": self.sensor_std.copy(),
        }

    def load_state(self, state: dict):
        self.W = state["W"].copy()
        self.b = state["b"].copy()
        self.sensor_mean = state["sensor_mean"].copy()
        self.sensor_std = state["sensor_std"].copy()
