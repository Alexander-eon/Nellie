"""
Фаза 3: коммуникация через среду.

Два агента в общем теле. Каждый видит ТОЛЬКО сенсоры тела
и своё прошлое действие. Действия складываются.

Никакого прямого канала между A и B.
Только физика тела.
"""

import time
from collections import deque
import numpy as np

from brain import Brain
from world_model import WorldModel
from body_sensor import BodySensor
from body_actuator import BodyActuator
from energy import Energy


class CommAgent:
    def __init__(
        self,
        dt: float = 0.05,
        brain_a_seed: int = 42,
        brain_b_seed: int = 99,
        model_a_seed: int = 44,
        model_b_seed: int = 45,
    ):
        self.dt = dt

        self.sensor = BodySensor()
        self.actuator = BodyActuator()
        self.energy = Energy(
            drain_rate=0.003,
            bounty_rate=0.003,
            bounty_probability=0.7,
            cpu_min=0.3, cpu_max=0.7,
            gpu_min=0.3, gpu_max=0.7,
            death_threshold=0.2,
            drift_period=500,
        )

        self.brain_a = Brain(seed=brain_a_seed)
        self.brain_b = Brain(seed=brain_b_seed)
        self.model_a = WorldModel(seed=model_a_seed)
        self.model_b = WorldModel(seed=model_b_seed)

        self.last_action_a = np.array([0.0, 0.0])
        self.last_action_b = np.array([0.0, 0.0])
        self.last_sensors = None

        self.step_count = 0
        self.alive = True

        # Логи
        self.errors_a = []
        self.errors_b = []
        self.energies = []
        self.actions_a = []
        self.actions_b = []
        self.sensors_log = []

    def start(self):
        self.actuator.start()

    def warmup(self, seconds: float = 2.0):
        """Прогревает GPU перед стартом обучения."""
        import time as _time
        self.actuator.set_intensity(0.3, 0.3)
        _time.sleep(seconds)
        self.actuator.set_intensity(0.0, 0.0)
        _time.sleep(0.5)

    def stop(self):
        self.actuator.stop()

    def step(self) -> dict:
        """Один такт: оба действуют через общее тело."""
        sensors, state = self.sensor.read()

        if self.last_sensors is None:
            self.last_sensors = sensors
            return {"step": 0, "energy": self.energy.energy, "alive": True}

        # --- A ---
        input_a = np.concatenate([sensors, self.last_action_a])
        action_a = self.brain_a.forward(input_a)

        # --- B ---
        input_b = np.concatenate([sensors, self.last_action_b])
        action_b = self.brain_b.forward(input_b)

        # --- Складываем действия ---
        final_action = 0.5 * action_a + 0.5 * action_b
        final_action = np.clip(final_action, 0.0, 1.0)

        # Применяем к телу
        self.actuator.set_intensity(final_action[0], final_action[1])
        time.sleep(self.dt)

        next_sensors, next_state = self.sensor.read()

        # --- Каждый учится предсказывать тело ---
        self.model_a.predict(self.last_sensors, action_a)
        error_a = self.model_a.update(next_sensors)
        reward_a = -error_a

        self.model_b.predict(self.last_sensors, action_b)
        error_b = self.model_b.update(next_sensors)
        reward_b = -error_b

        self.brain_a.update_weights(reward_a)
        self.brain_b.update_weights(reward_b)

        # --- Энергия ---
        self.energy.update(next_state)
        self.alive = self.energy.is_alive()

        # --- Запоминаем ---
        self.last_sensors = next_sensors
        self.last_action_a = action_a
        self.last_action_b = action_b
        self.step_count += 1

        # --- Логи ---
        self.errors_a.append(error_a)
        self.errors_b.append(error_b)
        self.energies.append(self.energy.energy)
        self.actions_a.append(action_a.copy())
        self.actions_b.append(action_b.copy())
        self.sensors_log.append(next_sensors.copy())

        return {
            "step": self.step_count,
            "error_a": error_a,
            "error_b": error_b,
            "energy": self.energy.energy,
            "a0": float(action_a[0]),
            "a1": float(action_a[1]),
            "b0": float(action_b[0]),
            "b1": float(action_b[1]),
            "alive": self.alive,
        }

    def mutual_information(self, x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
        if len(x) < 100:
            return 0.0
        hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
        p_xy = hist_2d / hist_2d.sum()
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
        return float(mi)

    def run(self, max_steps: int = 2000, log_every: int = 300):
        print(f"Старт: max {max_steps} тактов, dt={self.dt}s")
        print(f"{'шаг':>6} | {'err_A':>7} | {'err_B':>7} | "
              f"{'energy':>7} | {'a0':>6} | {'b0':>6} | {'a1':>6} | {'b1':>6}")
        print("-" * 76)

        self.warmup(2.0)
        for i in range(max_steps):
            log = self.step()
            if i % log_every == 0 and i > 0:
                print(f"{log['step']:>6} | "
                      f"{log['error_a']:>7.4f} | "
                      f"{log['error_b']:>7.4f} | "
                      f"{log['energy']:>7.3f} | "
                      f"{log['a0']:>6.3f} | "
                      f"{log['b0']:>6.3f} | "
                      f"{log['a1']:>6.3f} | "
                      f"{log['b1']:>6.3f}")

            if not log["alive"]:
                print()
                print(f"*** СМЕРТЬ на такте {log['step']} ***")
                break

        self._print_summary()

    def _print_summary(self):
        errors_a = np.array(self.errors_a)
        errors_b = np.array(self.errors_b)
        energies = np.array(self.energies)
        actions_a = np.array(self.actions_a)
        actions_b = np.array(self.actions_b)

        if len(errors_a) == 0:
            print("Нет данных.")
            return

        n = len(errors_a)
        head = min(100, n)
        tail = min(100, n)

        print()
        print("=" * 60)
        print(f"Прожито тактов: {self.step_count}")
        print(f"Ошибка A: первые {head} = {errors_a[:head].mean():.4f}, "
              f"последние {tail} = {errors_a[-tail:].mean():.4f}")
        print(f"Ошибка B: первые {head} = {errors_b[:head].mean():.4f}, "
              f"последние {tail} = {errors_b[-tail:].mean():.4f}")
        print(f"Средняя энергия: {energies.mean():.4f}")
        print(f"Финальная энергия: {energies[-1]:.4f}")

        if n > 100:
            corr_0 = np.corrcoef(actions_a[:, 0], actions_b[:, 0])[0, 1]
            corr_1 = np.corrcoef(actions_a[:, 1], actions_b[:, 1])[0, 1]
            print()
            print(f"Корреляция действий A[0] ↔ B[0]: {corr_0:.4f}")
            print(f"Корреляция действий A[1] ↔ B[1]: {corr_1:.4f}")

            mi = self.mutual_information(actions_a[:, 0], actions_b[:, 0])
            print(f"Взаимная информация A[0] ↔ B[0]: {mi:.4f}")

        print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=300)
    args = parser.parse_args()

    agent = CommAgent(dt=args.dt)
    agent.start()
    try:
        agent.run(max_steps=args.steps, log_every=args.log_every)
    finally:
        agent.stop()
        print("CommAgent остановлен.")
