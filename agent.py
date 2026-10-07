"""
Агент: связка мозга, модели мира, тела и энергии.

Версия 5: стохастический reward. Шум в награде создаёт "удачу".
Агент не может залипнуть — шум выбивает его из равновесия.
"""

import time
from collections import deque
import numpy as np

from brain import Brain
from world_model import WorldModel
from body_sensor import BodySensor
from body_actuator import BodyActuator
from energy import Energy
from checkpoint import save_checkpoint, load_checkpoint


class Agent:
    def __init__(
        self,
        dt: float = 0.1,
        curiosity_weight: float = 1.0,
        action_noise: float = 0.05,
        reward_noise: float = 0.1,
        bounty_probability: float = 0.7,
        history_len: int = 50,
        brain_seed: int = 42,
        model_seed: int = 43,
    ):
        self.dt = dt
        self.curiosity_weight = curiosity_weight
        self.action_noise = action_noise
        self.reward_noise = reward_noise
        self.bounty_probability = bounty_probability

        self.sensor = BodySensor()
        self.actuator = BodyActuator()
        self.brain = Brain(seed=brain_seed)
        self.world_model = WorldModel(seed=model_seed)
        self.energy = Energy(bounty_probability=bounty_probability)
        self.rng = np.random.default_rng(123)

        self.last_action = np.array([0.0, 0.0])
        self.last_sensors = None
        self.action_history = deque(maxlen=history_len)

        self.step_count = 0
        self.errors = []
        self.rewards = []
        self.curiosities = []
        self.energies = []
        self.alive = True

    def start(self):
        self.actuator.start()

    def stop(self):
        self.actuator.stop()

    def _novelty(self, action: np.ndarray) -> float:
        if len(self.action_history) < 5:
            return 0.5
        hist = np.array(self.action_history)
        mean = hist.mean(axis=0)
        dist_mean = np.linalg.norm(action - mean)
        dist_min = np.min(np.linalg.norm(hist - action, axis=1))
        return float(np.tanh(dist_mean * 3.0 + dist_min * 3.0))

    def step(self) -> dict:
        sensors, state = self.sensor.read()

        if self.last_sensors is None:
            self.last_sensors = sensors
            return {"step": 0, "reward": 0.0, "error": 0.0,
                    "curiosity": 0.0, "energy": self.energy.energy,
                    "cpu_int": 0.0, "gpu_int": 0.0, "alive": True}

        brain_input = np.concatenate([sensors, self.last_action])
        action = self.brain.forward(brain_input)

        noise = self.rng.normal(0.0, self.action_noise, 2)
        action = np.clip(action + noise, 0.0, 1.0)

        self.actuator.set_intensity(action[0], action[1])
        time.sleep(self.dt)

        next_sensors, next_state = self.sensor.read()

        self.world_model.predict(self.last_sensors, action)
        error = self.world_model.update(next_sensors)

        novelty = self._novelty(action)

        # Стохастический reward
        rng_noise = self.rng.normal(0.0, self.reward_noise)
        reward = -error + self.curiosity_weight * novelty + rng_noise

        self.energy.update(next_state)

        self.action_history.append(action.copy())
        self.brain.update_weights(reward)

        self.alive = self.energy.is_alive()

        self.last_sensors = next_sensors
        self.last_action = action
        self.step_count += 1

        self.errors.append(error)
        self.rewards.append(reward)
        self.curiosities.append(novelty)
        self.energies.append(self.energy.energy)

        return {
            "step": self.step_count,
            "reward": reward,
            "error": error,
            "curiosity": novelty,
            "energy": self.energy.energy,
            "cpu_int": float(action[0]),
            "gpu_int": float(action[1]),
            "alive": self.alive,
        }

    def run(self, max_steps: int = 5000, log_every: int = 100):
        print(f"Старт: max {max_steps} тактов, dt={self.dt}s")
        print(f"Энергия старта: {self.energy.energy:.3f}")
        print(f"Шум reward: {self.reward_noise}")
        print(f"{'шаг':>6} | {'reward':>8} | {'error':>8} | "
              f"{'curio':>7} | {'energy':>7} | {'cpu':>6} | {'gpu':>6}")
        print("-" * 72)

        for i in range(max_steps):
            log = self.step()

            if i % log_every == 0 and i > 0:
                print(f"{log['step']:>6} | "
                      f"{log['reward']:>8.4f} | "
                      f"{log['error']:>8.4f} | "
                      f"{log['curiosity']:>7.3f} | "
                      f"{log['energy']:>7.3f} | "
                      f"{log['cpu_int']:>6.3f} | "
                      f"{log['gpu_int']:>6.3f}")

            if not log["alive"]:
                print()
                print(f"*** СМЕРТЬ на такте {log['step']} ***")
                break

        self._print_summary()

    def save(self, path: str = "checkpoints/brain_latest.npz"):
        """Сохраняет мозг и модель мира."""
        return save_checkpoint(self.brain, self.world_model, path)

    def load(self, path: str = "checkpoints/brain_latest.npz"):
        """Загружает мозг и модель мира."""
        return load_checkpoint(self.brain, self.world_model, path)

    def _print_summary(self):
        errors = np.array(self.errors)
        energies = np.array(self.energies)

        if len(errors) == 0:
            print("Нет данных.")
            return

        print()
        print("=" * 50)
        print(f"Прожито тактов: {self.step_count}")
        print(f"Средняя ошибка (первые 100):  {errors[:100].mean():.4f}")
        if len(errors) > 100:
            print(f"Средняя ошибка (последние 100): {errors[-100:].mean():.4f}")
        print(f"Средняя энергия: {energies.mean():.4f}")
        print(f"Финальная энергия: {energies[-1]:.4f}")
        print("=" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--curiosity", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--reward-noise", type=float, default=0.1)
    parser.add_argument("--bounty-prob", type=float, default=0.7)
    args = parser.parse_args()

    agent = Agent(
        dt=args.dt,
        curiosity_weight=args.curiosity,
        action_noise=args.noise,
        reward_noise=args.reward_noise,
        bounty_probability=args.bounty_prob,
    )
    agent.start()
    try:
        agent.run(max_steps=args.max_steps, log_every=args.log_every)
    finally:
        agent.stop()
        print("Агент остановлен.")
