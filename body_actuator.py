"""
Актуатор тела: нагрузка на CPU и GPU через отдельные ПРОЦЕССЫ.
Версия 2: CPU-нагрузка через пул процессов (по одному на ядро).
"""

import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time
import psutil


def cpu_worker(shm_name: str, stop_event, worker_id: int):
    """Один CPU-воркер. Грузит одно ядро."""
    shm = shared_memory.SharedMemory(name=shm_name)
    intensities = np.ndarray((2,), dtype=np.float64, buffer=shm.buf)

    while not stop_event.is_set():
        target = float(intensities[0])
        if target < 0.01:
            time.sleep(0.02)
            continue
        period = 0.05
        work_time = period * target
        t0 = time.time()
        x = 0.0
        while time.time() - t0 < work_time:
            for i in range(100000):
                x += i * 0.5
        rest = period - work_time
        if rest > 0:
            time.sleep(rest)
    shm.close()


def gpu_worker(shm_name: str, stop_event):
    """GPU-воркер. Читает интенсивность, грузит CUDA."""
    import cupy as cp
    shm = shared_memory.SharedMemory(name=shm_name)
    intensities = np.ndarray((2,), dtype=np.float64, buffer=shm.buf)

    mat_size = 2048
    a = cp.random.rand(mat_size, mat_size).astype(cp.float32)
    b = cp.random.rand(mat_size, mat_size).astype(cp.float32)

    while not stop_event.is_set():
        target = float(intensities[1])
        if target < 0.01:
            time.sleep(0.02)
            continue
        work_time = 0.03 * target
        t0 = time.time()
        while time.time() - t0 < work_time:
            _ = cp.matmul(a, b)
            cp.cuda.Stream.null.synchronize()
        rest = 0.03 - work_time
        if rest > 0:
            time.sleep(rest)

    del a, b
    cp.get_default_memory_pool().free_all_blocks()
    shm.close()


class BodyActuator:
    def __init__(self, n_cpu_workers: int = None):
        if n_cpu_workers is None:
            n_cpu_workers = max(1, psutil.cpu_count(logical=False) or 8)
        self.n_cpu_workers = n_cpu_workers

        self.shm = shared_memory.SharedMemory(create=True, size=2 * 8)
        self.intensities = np.ndarray((2,), dtype=np.float64, buffer=self.shm.buf)
        self.intensities[:] = 0.0

        self.stop_event = mp.Event()
        self.cpu_procs = []
        self.gpu_proc = None

    def start(self):
        for i in range(self.n_cpu_workers):
            p = mp.Process(
                target=cpu_worker,
                args=(self.shm.name, self.stop_event, i),
                daemon=True,
            )
            p.start()
            self.cpu_procs.append(p)

        self.gpu_proc = mp.Process(
            target=gpu_worker,
            args=(self.shm.name, self.stop_event),
            daemon=True,
        )
        self.gpu_proc.start()

    def set_intensity(self, cpu: float, gpu: float):
        self.intensities[0] = float(np.clip(cpu, 0.0, 1.0))
        self.intensities[1] = float(np.clip(gpu, 0.0, 1.0))

    def stop(self):
        self.stop_event.set()
        for p in self.cpu_procs:
            p.join(timeout=2.0)
        if self.gpu_proc:
            self.gpu_proc.join(timeout=2.0)
        self.shm.close()
        self.shm.unlink()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from body_sensor import BodySensor

    sensor = BodySensor()
    actuator = BodyActuator()
    print(f"CPU воркеров: {actuator.n_cpu_workers}")
    print()
    actuator.start()

    print("Тест: 4 сценария по 3 секунды")
    print()

    scenarios = [
        ("покой",        0.0, 0.0),
        ("CPU 50%",      0.5, 0.0),
        ("GPU 50%",      0.0, 0.5),
        ("CPU+GPU 100%", 1.0, 1.0),
    ]

    for label, cpu_i, gpu_i in scenarios:
        actuator.set_intensity(cpu_i, gpu_i)
        print(f"--- {label} ---")
        time.sleep(2.0)
        for j in range(3):
            vec, state = sensor.read()
            print(f"  cpu_load={state['cpu_load']:.3f}  "
                  f"cpu_freq={state['cpu_freq_norm']:.3f}  "
                  f"gpu_load={state['gpu_load']:.3f}  "
                  f"gpu_temp={state['gpu_temp']:.3f}  "
                  f"gpu_power={state['gpu_power']:.3f}")
            time.sleep(0.3)
        print()

    actuator.stop()
    print("Готово.")
