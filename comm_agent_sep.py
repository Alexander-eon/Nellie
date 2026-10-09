"""
Разделённое зрение: A знает target_gpu, B знает target_cpu.

A управляет CPU. B управляет GPU.
A знает target_gpu (но не может двигать GPU).
B знает target_cpu (но не может двигать CPU).

Energy: bounty если CPU в зоне target_cpu И GPU в зоне target_gpu.

Без коммуникации — никто не выживет.
"""

import time
import numpy as np

from brain import Brain
from world_model import WorldModel
from body_sensor import BodySensor
from body_actuator import BodyActuator
from energy import Energy


class CommAgentSep:
    def __init__(
        self,
        dt: float = 0.05,
        brain_a_seed: int = 42,
        brain_b_seed: int = 99,
        model_a_seed: int = 44,
        model_b_seed: int = 45,
        target_change_period: int = 200,
        zone_width: float = 0.25,
    ):
        self.dt = dt
        self.target_change_period = target_change_period
        self.zone_width = zone_width

        self.sensor = BodySensor()
        self.actuator = BodyActuator()
        self.energy = Energy(
            drain_rate=0.002,
            bounty_rate=0.004,
            bounty_probability=0.7,
            cpu_min=0.3, cpu_max=0.7,
            gpu_min=0.3, gpu_max=0.7,
            death_threshold=0.2,
            drift_period=100000,  # отключаем дрейф - у нас свои цели
            connection_penalty=0.5,
        )

        # A: вход = [cpu_load, gpu_load, target_gpu, last_action_a] = 4
        self.brain_a = Brain(n_input=4, seed=brain_a_seed)
        # B: вход = [gpu_load, cpu_load, target_cpu, last_action_b] = 4
        self.brain_b = Brain(n_input=4, seed=brain_b_seed)

        self.model_a = WorldModel(seed=model_a_seed)
        self.model_b = WorldModel(seed=model_b_seed)

        self.last_action_a = np.array([0.0])
        self.last_action_b = np.array([0.0])
        self.last_cpu = None
        self.last_gpu = None

        # Цели
        self.target_cpu = 0.5
        self.target_gpu = 0.5
        self.rng = np.random.default_rng(42)

        self.step_count = 0
        self.alive = True

        # Логи
        self.errors_a = []
        self.errors_b = []
        self.energies = []
        self.actions_a = []
        self.actions_b = []
        self.cpu_log = []
        self.gpu_log = []
        self.target_cpu_log = []
        self.target_gpu_log = []

    def start(self):
        self.actuator.start()

    def warmup(self, seconds: float = 2.0):
        self.actuator.set_intensity(0.3, 0.3)
        time.sleep(seconds)
        self.actuator.set_intensity(0.0, 0.0)
        time.sleep(0.5)

    def stop(self):
        self.actuator.stop()

    def _update_targets(self):
        """Меняет target_cpu и target_gpu каждые N тактов."""
        if self.step_count % self.target_change_period == 0:
            self.target_cpu = float(self.rng.uniform(0.35, 0.65))
            self.target_gpu = float(self.rng.uniform(0.35, 0.65))

    def step(self) -> dict:
        """Один такт: A действует на CPU, B — на GPU."""
        sensors, state = self.sensor.read()
        cpu_load = state.get("cpu_load", 0.0)
        gpu_load = state.get("gpu_load", 0.0)

        if self.last_cpu is None:
            self.last_cpu = cpu_load
            self.last_gpu = gpu_load
            return {"step": 0, "energy": self.energy.energy, "alive": True}

        # Меняем цели
        self._update_targets()

        # A: видит ОБА ресурса + цель для напарника + своё прошлое действие
        input_a = np.array([cpu_load, gpu_load, self.target_gpu, self.last_action_a[0]])
        action_a = self.brain_a.forward(input_a)
        cpu_intensity = float(action_a[0])

        # B: видит ОБА ресурса + цель для напарника + своё прошлое действие
        input_b = np.array([gpu_load, cpu_load, self.target_cpu, self.last_action_b[0]])
        action_b = self.brain_b.forward(input_b)
        gpu_intensity = float(action_b[0])

        # Применяем оба действия
        self.actuator.set_intensity(cpu_intensity, gpu_intensity)
        time.sleep(self.dt)

        next_sensors, next_state = self.sensor.read()
        next_cpu = next_state.get("cpu_load", 0.0)
        next_gpu = next_state.get("gpu_load", 0.0)

        # Командный reward: оба за общий результат
        # A знает target_gpu, B знает target_cpu
        # Никто не может достичь своей цели сам
        # Только через коммуникацию
        error_cpu = abs(next_cpu - self.target_cpu)
        error_gpu = abs(next_gpu - self.target_gpu)
        reward_team = -(error_cpu + error_gpu)

        self.brain_a.update_weights(reward_team)
        self.brain_b.update_weights(reward_team)

        # Для логов
        error_a = error_cpu
        error_b = error_gpu

        # Energy: bounty если cpu в зоне target_cpu И gpu в зоне target_gpu
        half = self.zone_width / 2.0
        cpu_in = (self.target_cpu - half) <= next_cpu <= (self.target_cpu + half)
        gpu_in = (self.target_gpu - half) <= next_gpu <= (self.target_gpu + half)

        # Обновляем зоны в energy под текущие цели
        self.energy.cpu_min = self.target_cpu - half
        self.energy.cpu_max = self.target_cpu + half
        self.energy.gpu_min = self.target_gpu - half
        self.energy.gpu_max = self.target_gpu + half

        self.energy.update(next_state)
        self.alive = self.energy.is_alive()

        # Запоминаем
        self.last_cpu = next_cpu
        self.last_gpu = next_gpu
        self.last_action_a = action_a
        self.last_action_b = action_b
        self.step_count += 1

        # Логи
        self.errors_a.append(error_a)
        self.errors_b.append(error_b)
        self.energies.append(self.energy.energy)
        self.actions_a.append(action_a.copy())
        self.actions_b.append(action_b.copy())
        self.cpu_log.append(next_cpu)
        self.gpu_log.append(next_gpu)
        self.target_cpu_log.append(self.target_cpu)
        self.target_gpu_log.append(self.target_gpu)

        return {
            "step": self.step_count,
            "error_a": error_a,
            "error_b": error_b,
            "energy": self.energy.energy,
            "cpu": next_cpu,
            "gpu": next_gpu,
            "target_cpu": self.target_cpu,
            "target_gpu": self.target_gpu,
            "a": cpu_intensity,
            "b": gpu_intensity,
            "cpu_in": cpu_in,
            "gpu_in": gpu_in,
            "alive": self.alive,
        }

    def run(self, max_steps: int = 5000, log_every: int = 500):
        print(f"Старт: max {max_steps} тактов, dt={self.dt}s")
        print(f"Цели меняются каждые {self.target_change_period} тактов")
        print(f"Ширина зоны: {self.zone_width}")
        print(f"{'шаг':>6} | {'err_A':>7} | {'err_B':>7} | "
              f"{'energy':>7} | {'cpu':>6} | {'tgt_c':>6} | "
              f"{'gpu':>6} | {'tgt_g':>6} | {'in':>4}")
        print("-" * 80)

        self.warmup(2.0)

        for i in range(max_steps):
            log = self.step()
            if i % log_every == 0 and i > 0:
                both = "OK" if (log.get("cpu_in") and log.get("gpu_in")) else "NO"
                print(f"{log['step']:>6} | "
                      f"{log.get('error_a', 0.0):>7.4f} | "
                      f"{log.get('error_b', 0.0):>7.4f} | "
                      f"{log['energy']:>7.3f} | "
                      f"{log.get('cpu', 0.0):>6.3f} | "
                      f"{log.get('target_cpu', 0.0):>6.3f} | "
                      f"{log.get('gpu', 0.0):>6.3f} | "
                      f"{log.get('target_gpu', 0.0):>6.3f} | "
                      f"{both:>4}")

            if not log.get("alive", True):
                print()
                print(f"*** СМЕРТЬ на такте {log['step']} ***")
                break

        self._print_summary()

    def _print_summary(self):
        if len(self.energies) == 0:
            print("Нет данных.")
            return

        n = len(self.energies)
        head = min(100, n)
        tail = min(100, n)

        print()
        print("=" * 60)
        print(f"Прожито тактов: {self.step_count}")
        print(f"Средняя энергия: {np.mean(self.energies):.4f}")
        print(f"Финальная энергия: {self.energies[-1]:.4f}")

        # Сохраняем для TE
        np.savez(
            "/tmp/sep_data.npz",
            actions_a=np.array(self.actions_a),
            actions_b=np.array(self.actions_b),
            cpu=np.array(self.cpu_log),
            gpu=np.array(self.gpu_log),
            target_cpu=np.array(self.target_cpu_log),
            target_gpu=np.array(self.target_gpu_log),
        )
        print("Данные сохранены в /tmp/sep_data.npz")
        print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()

    agent = CommAgentSep(dt=args.dt)
    agent.start()
    try:
        agent.run(max_steps=args.steps, log_every=args.log_every)
    finally:
        agent.stop()
        print("CommAgentSep остановлен.")
