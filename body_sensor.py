"""
Сенсоры тела компьютера.

Читает:
- CPU: загрузка, частота, температура (если доступна)
- RAM: использование
- GPU: температура, загрузка, потребление, память, вентилятор
- Диск: активность
- Сеть: трафик

Возвращает вектор float-значений — вход для мозга.
"""

import subprocess
import psutil
import numpy as np


class BodySensor:
    def __init__(self):
        self.n_cpu = psutil.cpu_count()
        self.prev_disk = None
        self.prev_net = None

        # Проверяем, доступна ли nvidia-smi
        self.has_gpu = self._check_nvidia_smi()

        # Кэш предыдущих значений для нормировки
        self.cpu_freq_max = self._get_cpu_freq_max()

    def _check_nvidia_smi(self) -> bool:
        try:
            subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, check=True, timeout=2,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _get_cpu_freq_max(self) -> float:
        try:
            freq = psutil.cpu_freq()
            return freq.max if freq and freq.max > 0 else 5000.0
        except Exception:
            return 5000.0

    def _gpu_query(self, fields: str) -> list:
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True, timeout=2,
            )
            return out.stdout.strip().split("\n")
        except Exception:
            return []

    def read_cpu(self) -> dict:
        cpu_percent = psutil.cpu_percent(interval=None) / 100.0
        freq = psutil.cpu_freq()
        freq_cur = freq.current if freq else 0.0
        freq_norm = freq_cur / self.cpu_freq_max if self.cpu_freq_max > 0 else 0.0
        return {
            "cpu_load": cpu_percent,
            "cpu_freq_norm": freq_norm,
        }

    def read_memory(self) -> dict:
        mem = psutil.virtual_memory()
        return {
            "mem_used": mem.percent / 100.0,
        }

    def read_gpu(self) -> dict:
        if not self.has_gpu:
            return {
                "gpu_temp": 0.0,
                "gpu_load": 0.0,
                "gpu_power": 0.0,
                "gpu_mem": 0.0,
                "gpu_fan": 0.0,
            }

        lines = self._gpu_query(
            "temperature.gpu,utilization.gpu,power.draw,memory.used,fan.speed"
        )
        if not lines:
            return {
                "gpu_temp": 0.0,
                "gpu_load": 0.0,
                "gpu_power": 0.0,
                "gpu_mem": 0.0,
                "gpu_fan": 0.0,
            }

        parts = [p.strip() for p in lines[0].split(",")]
        try:
            temp = float(parts[0])
            load = float(parts[1])
            power = float(parts[2])
            mem = float(parts[3])
            fan = float(parts[4])
        except (ValueError, IndexError):
            temp = load = power = mem = fan = 0.0

        return {
            "gpu_temp": temp / 100.0,
            "gpu_load": load / 100.0,
            "gpu_power": power / 400.0,
            "gpu_mem": mem / 24576.0,
            "gpu_fan": fan / 100.0,
        }

    def read_disk(self) -> dict:
        io = psutil.disk_io_counters()
        if io is None:
            return {"disk_read": 0.0, "disk_write": 0.0}

        cur = (io.read_bytes, io.write_bytes)
        if self.prev_disk is None:
            self.prev_disk = cur
            return {"disk_read": 0.0, "disk_write": 0.0}

        d_read = cur[0] - self.prev_disk[0]
        d_write = cur[1] - self.prev_disk[1]
        self.prev_disk = cur

        # Нормировка: 100 МБ/такт = 1.0
        return {
            "disk_read": min(d_read / 1e8, 1.0),
            "disk_write": min(d_write / 1e8, 1.0),
        }

    def read_net(self) -> dict:
        net = psutil.net_io_counters()
        cur = (net.bytes_sent, net.bytes_recv)
        if self.prev_net is None:
            self.prev_net = cur
            return {"net_sent": 0.0, "net_recv": 0.0}

        d_sent = cur[0] - self.prev_net[0]
        d_recv = cur[1] - self.prev_net[1]
        self.prev_net = cur

        # Нормировка: 10 МБ/такт = 1.0
        return {
            "net_sent": min(d_sent / 1e7, 1.0),
            "net_recv": min(d_recv / 1e7, 1.0),
        }

    def read(self) -> np.ndarray:
        """
        Читает все сенсоры и возвращает вектор float64.
        """
        state = {}
        state.update(self.read_cpu())
        state.update(self.read_memory())
        state.update(self.read_gpu())
        state.update(self.read_disk())
        state.update(self.read_net())

        keys = [
            "cpu_load", "cpu_freq_norm", "mem_used",
            "gpu_temp", "gpu_load", "gpu_power", "gpu_mem", "gpu_fan",
            "disk_read", "disk_write", "net_sent", "net_recv",
        ]
        vec = np.array([state[k] for k in keys], dtype=np.float64)
        return vec, state


if __name__ == "__main__":
    import time

    sensor = BodySensor()
    print(f"CPU cores: {sensor.n_cpu}")
    print(f"GPU detected: {sensor.has_gpu}")
    print()

    for i in range(3):
        vec, state = sensor.read()
        print(f"=== Чтение {i+1} ===")
        for k, v in state.items():
            print(f"  {k:15s} = {v:.4f}")
        print(f"  Вектор: {vec}")
        print()
        time.sleep(1)
