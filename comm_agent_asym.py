"""
Этап 1: асимметрия информации.

A знает скрытую переменную (hidden), B — нет.
hidden влияет на зону bounty.
A должен передать hidden через свои действия.
B должен научиться его читать.

Вопрос: вырастет ли transfer entropy A→B?
"""

import time
from collections import deque
import numpy as np

from brain import Brain
from world_model import WorldModel
from body_sensor import BodySensor
from body_actuator import BodyActuator
from energy import Energy


class CommAgentAsym:
    def __init__(
        self,
        dt: float = 0.05,
        brain_a_seed: int = 42,
        brain_b_seed: int = 99,
        model_a_seed: int = 44,
        model_b_seed: int = 45,
        hidden_change_period: int = 100,
        hidden_reward_weight: float = 0.5,
    ):
        self.dt = dt
        self.hidden_change_period = hidden_change_period
        self.hidden_reward_weight = hidden_reward_weight

        self.sensor = BodySensor()
        self.actuator = BodyActuator()
        self.energy = Energy(
            drain_rate=0.002,
            bounty_rate=0.004,
            bounty_probability=0.7,
            cpu_min=0.3, cpu_max=0.7,
            gpu_min=0.3, gpu_max=0.7,
            death_threshold=0.2,
            drift_period=500,
            connection_penalty=1.5,  # мягче, чем в жёстком режиме
        )

        # A получает 15 входов: 12 сенсоров + 2 действия + hidden
        self.brain_a = Brain(n_input=15, seed=brain_a_seed)
        # B получает 14 входов: 12 сенсоров + 2 действия
        self.brain_b = Brain(n_input=14, seed=brain_b_seed)

        self.model_a = WorldModel(seed=model_a_seed)
        self.model_b = WorldModel(seed=model_b_seed)

        self.last_action_a = np.array([0.0, 0.0])
        self.last_action_b = np.array([0.0, 0.0])
        self.last_sensors_a = None
        self.last_sensors_b = None

        # Скрытая переменная
        self.hidden = 0.5
        self.rng = np.random.default_rng(42)
        self.step_count = 0
        self.alive = True

        self.errors_a = []
        self.errors_b = []
        self.energies = []
        self.actions_a = []
        self.actions_b = []
        self.sensors_log = []
        self.hidden_log = []
        self.recent_b = deque(maxlen=100)
        self.recent_hidden = deque(maxlen=100)

    def start(self):
        self.actuator.start()

    def warmup(self, seconds: float = 2.0):
        self.actuator.set_intensity(0.3, 0.3)
        time.sleep(seconds)
        self.actuator.set_intensity(0.0, 0.0)
        time.sleep(0.5)

    def stop(self):
        self.actuator.stop()

    def _update_hidden(self):
        """Меняет hidden каждые N тактов."""
        if self.step_count % self.hidden_change_period == 0:
            self.hidden = float(self.rng.random())
        # Применяем hidden к зоне bounty
        # hidden ∈ [0, 1] -> zone_offset ∈ [-0.3, +0.3]
        # Усилено: без учёта hidden агент вылетает из зоны
        self.energy.zone_offset = (self.hidden - 0.5) * 0.4

    def step(self) -> dict:
        """Один такт: A знает hidden, B — нет."""
        sensors, state = self.sensor.read()

        if self.last_sensors_a is None:
            self.last_sensors_a = sensors
            self.last_sensors_b = sensors
            return {"step": 0, "energy": self.energy.energy, "alive": True}

        # Обновляем hidden
        self._update_hidden()

        # === A: действует ===
        # A видит: сенсоры + своё прошлое действие + hidden
        input_a = np.concatenate([sensors, self.last_action_a, [self.hidden]])
        action_a = self.brain_a.forward(input_a)

        self.actuator.set_intensity(action_a[0], action_a[1])
        time.sleep(self.dt / 2.0)

        sensors_mid, state_mid = self.sensor.read()

        # A учится предсказывать тело
        self.model_a.predict(self.last_sensors_a, action_a)
        error_a = self.model_a.update(sensors_mid)
        self.brain_a.update_weights(-error_a)

        # === B: действует ===
        # B видит: сенсоры + своё прошлое действие (hidden НЕ видит)
        input_b = np.concatenate([sensors_mid, self.last_action_b])
        action_b = self.brain_b.forward(input_b)

        self.actuator.set_intensity(action_b[0], action_b[1])
        time.sleep(self.dt / 2.0)

        sensors_end, state_end = self.sensor.read()

        # B учится предсказывать тело
        self.model_b.predict(self.last_sensors_b, action_b)
        error_b = self.model_b.update(sensors_end)

        # Добавляем в буферы
        self.recent_b.append(action_b[0])
        self.recent_hidden.append(self.hidden)

        # Бонус за корреляцию действий B с hidden
        hidden_bonus = 0.0
        if len(self.recent_b) >= 100:
            arr_b = np.array(self.recent_b)
            arr_h = np.array(self.recent_hidden)
            if arr_b.std() > 1e-6 and arr_h.std() > 1e-6:
                corr = np.corrcoef(arr_b, arr_h)[0, 1]
                if not np.isnan(corr):
                    hidden_bonus = abs(corr)

        # Reward B = -error_b + бонус за корреляцию с hidden
        reward_b = -error_b + self.hidden_reward_weight * hidden_bonus
        self.brain_b.update_weights(reward_b)

        # Энергия (зона уже сдвинута через zone_offset)
        self.energy.update(state_end)
        self.alive = self.energy.is_alive()

        # Запоминаем
        self.last_sensors_a = sensors_end
        self.last_sensors_b = sensors_mid
        self.last_action_a = action_a
        self.last_action_b = action_b
        self.step_count += 1

        # Логи
        self.errors_a.append(error_a)
        self.errors_b.append(error_b)
        self.energies.append(self.energy.energy)
        self.actions_a.append(action_a.copy())
        self.actions_b.append(action_b.copy())
        self.sensors_log.append(sensors_end.copy())
        self.hidden_log.append(self.hidden)

        return {
            "step": self.step_count,
            "error_a": error_a,
            "error_b": error_b,
            "energy": self.energy.energy,
            "a0": float(action_a[0]),
            "a1": float(action_a[1]),
            "b0": float(action_b[0]),
            "b1": float(action_b[1]),
            "hidden": self.hidden,
            "alive": self.alive,
        }

    def run(self, max_steps: int = 5000, log_every: int = 500):
        print(f"Старт: max {max_steps} тактов, dt={self.dt}s")
        print(f"hidden меняется каждые {self.hidden_change_period} тактов")
        print(f"{'шаг':>6} | {'err_A':>7} | {'err_B':>7} | "
              f"{'energy':>7} | {'hidden':>7} | {'a0':>6} | {'b0':>6}")
        print("-" * 68)

        self.warmup(2.0)

        for i in range(max_steps):
            log = self.step()
            if i % log_every == 0 and i > 0:
                print(f"{log['step']:>6} | "
                      f"{log.get('error_a', 0.0):>7.4f} | "
                      f"{log.get('error_b', 0.0):>7.4f} | "
                      f"{log['energy']:>7.3f} | "
                      f"{log.get('hidden', 0.0):>7.3f} | "
                      f"{log.get('a0', 0.0):>6.3f} | "
                      f"{log.get('b0', 0.0):>6.3f}")

            if not log.get("alive", True):
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

        # Сохраняем данные для TE
        np.savez(
            "/tmp/asym_data.npz",
            actions_a=actions_a,
            actions_b=actions_b,
            hidden=np.array(self.hidden_log),
        )
        print("Данные сохранены в /tmp/asym_data.npz")
        print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()

    agent = CommAgentAsym(dt=args.dt)
    agent.start()
    try:
        agent.run(max_steps=args.steps, log_every=args.log_every)
    finally:
        agent.stop()
        print("CommAgentAsym остановлен.")
