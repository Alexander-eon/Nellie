"""
Энергия агента. Основа жизни.

Версия 3: усложнённое выживание.
- Узкая зона bounty (0.3..0.7).
- Большой drain (0.005).
- Малый bounty (0.002).
- Дрейф цели: каждые drift_period тактов зона сдвигается.
"""

import numpy as np


class Energy:
    def __init__(
        self,
        initial: float = 0.5,
        drain_rate: float = 0.002,
        bounty_rate: float = 0.002,
        bounty_probability: float = 0.7,
        death_threshold: float = 0.35,
        energy_max: float = 0.9,
        connection_penalty: float = 1.5,  # множитель drain при слабой связи
        cpu_min: float = 0.45,
        cpu_max: float = 0.55,
        gpu_min: float = 0.45,
        gpu_max: float = 0.55,
        temp_max: float = 0.75,
        drift_period: int = 300,
        drift_range: float = 0.1,
        seed: int = 44,
    ):
        self.initial = initial
        self.energy = initial
        self.drain_rate = drain_rate
        self.bounty_rate = bounty_rate
        self.bounty_probability = bounty_probability
        self.death_threshold = death_threshold
        self.energy_max = energy_max
        self.connection_penalty = connection_penalty
        self.connection_strength = 1.0  # 0..1, 1 = сильная связь

        self.cpu_min_base = cpu_min
        self.cpu_max_base = cpu_max
        self.gpu_min_base = gpu_min
        self.gpu_max_base = gpu_max
        self.temp_max = temp_max

        # Дрейф цели
        self.drift_period = drift_period
        self.drift_range = drift_range
        self.cpu_min = cpu_min
        self.cpu_max = cpu_max
        self.gpu_min = gpu_min
        self.gpu_max = gpu_max

        self.rng = np.random.default_rng(seed)

        self.history = []
        self.deaths = 0
        self.luck_events = 0
        self.drift_count = 0

    def _roll_bounty(self) -> float:
        if self.rng.random() < self.bounty_probability:
            return +self.bounty_rate
        else:
            self.luck_events += 1
            return -self.bounty_rate * 0.3

    def _maybe_drift(self):
        """Сдвигает целевую зону каждые drift_period тактов."""
        if len(self.history) % self.drift_period == 0 and len(self.history) > 0:
            shift_cpu = self.rng.uniform(-self.drift_range, self.drift_range)
            shift_gpu = self.rng.uniform(-self.drift_range, self.drift_range)

            self.cpu_min = float(np.clip(self.cpu_min_base + shift_cpu, 0.1, 0.8))
            self.cpu_max = float(np.clip(self.cpu_max_base + shift_cpu, 0.2, 0.9))
            self.gpu_min = float(np.clip(self.gpu_min_base + shift_gpu, 0.1, 0.8))
            self.gpu_max = float(np.clip(self.gpu_max_base + shift_gpu, 0.2, 0.9))

            self.drift_count += 1

    def update(self, state: dict) -> float:
        self._maybe_drift()

        # Drain зависит от связи между агентами
        # connection_strength ∈ [0, 1]
        # 0 = нет связи, 1 = сильная связь
        drain_multiplier = 1.0 + (1.0 - self.connection_strength) * self.connection_penalty
        delta = -self.drain_rate * drain_multiplier

        cpu = state.get("cpu_load", 0.0)
        gpu = state.get("gpu_load", 0.0)
        gpu_temp = state.get("gpu_temp", 0.0)

        if self.cpu_min <= cpu <= self.cpu_max:
            delta += self._roll_bounty()

        if self.gpu_min <= gpu <= self.gpu_max:
            delta += self._roll_bounty()

        if gpu_temp > self.temp_max:
            overheat = (gpu_temp - self.temp_max) / (1.0 - self.temp_max)
            delta -= self.bounty_rate * overheat * 5.0

        if cpu < 0.05 and gpu < 0.05:
            delta -= self.bounty_rate

        self.energy = float(np.clip(self.energy + delta, 0.0, 1.0))
        self.history.append(self.energy)

        return delta

    def is_alive(self) -> bool:
        """Агент жив, пока энергия выше порога смерти."""
        return self.energy > self.death_threshold

    def reset(self):
        self.energy = self.initial
        self.deaths += 1

    def get_state(self) -> dict:
        return {
            "energy": self.energy,
            "deaths": self.deaths,
            "luck_events": self.luck_events,
            "drift_count": self.drift_count,
        }

    def load_state(self, state: dict):
        self.energy = state["energy"]
        self.deaths = state["deaths"]
        self.luck_events = state.get("luck_events", 0)
        self.drift_count = state.get("drift_count", 0)
