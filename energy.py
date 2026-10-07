"""
Энергия агента. Основа жизни.

Версия 2: стохастический bounty.
Если тело в норме — с вероятностью p получаешь bounty.
С вероятностью (1-p) — штраф. Это "не повезло".
"""

import numpy as np


class Energy:
    def __init__(
        self,
        initial: float = 0.5,
        drain_rate: float = 0.002,
        bounty_rate: float = 0.003,
        bounty_probability: float = 0.7,
        cpu_min: float = 0.15,
        cpu_max: float = 0.85,
        gpu_min: float = 0.15,
        gpu_max: float = 0.90,
        temp_max: float = 0.75,
        seed: int = 44,
    ):
        self.initial = initial
        self.energy = initial
        self.drain_rate = drain_rate
        self.bounty_rate = bounty_rate
        self.bounty_probability = bounty_probability

        self.cpu_min = cpu_min
        self.cpu_max = cpu_max
        self.gpu_min = gpu_min
        self.gpu_max = gpu_max
        self.temp_max = temp_max

        self.rng = np.random.default_rng(seed)

        self.history = []
        self.deaths = 0
        self.luck_events = 0  # счётчик "повезло/не повезло"

    def _roll_bounty(self) -> float:
        """
        Бросает монету: с вероятностью p — bounty, иначе — штраф.
        Возвращает +bounty_rate или -bounty_rate.
        """
        if self.rng.random() < self.bounty_probability:
            return +self.bounty_rate
        else:
            self.luck_events += 1
            return -self.bounty_rate * 0.3

    def update(self, state: dict) -> float:
        """
        Обновляет энергию на основе состояния тела.
        Возвращает изменение энергии за такт.
        """
        delta = -self.drain_rate

        cpu = state.get("cpu_load", 0.0)
        gpu = state.get("gpu_load", 0.0)
        gpu_temp = state.get("gpu_temp", 0.0)

        # CPU в норме — бросаем монету
        if self.cpu_min <= cpu <= self.cpu_max:
            delta += self._roll_bounty()

        # GPU в норме — бросаем монету
        if self.gpu_min <= gpu <= self.gpu_max:
            delta += self._roll_bounty()

        # Перегрев — чистый штраф, без монеты
        if gpu_temp > self.temp_max:
            overheat = (gpu_temp - self.temp_max) / (1.0 - self.temp_max)
            delta -= self.bounty_rate * overheat * 5.0

        # Полный простой — штраф
        if cpu < 0.05 and gpu < 0.05:
            delta -= self.bounty_rate

        self.energy = float(np.clip(self.energy + delta, 0.0, 1.0))
        self.history.append(self.energy)

        return delta

    def is_alive(self) -> bool:
        return self.energy > 0.0

    def reset(self):
        self.energy = self.initial
        self.deaths += 1

    def get_state(self) -> dict:
        return {
            "energy": self.energy,
            "deaths": self.deaths,
            "luck_events": self.luck_events,
        }

    def load_state(self, state: dict):
        self.energy = state["energy"]
        self.deaths = state["deaths"]
        self.luck_events = state.get("luck_events", 0)
