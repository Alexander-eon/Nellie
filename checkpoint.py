"""
Чекпоинты: сохранение и загрузка состояния мозга и модели мира.
"""

import os
import numpy as np


def save_checkpoint(
    brain,
    world_model=None,
    path: str = "checkpoints/brain_latest.npz",
    metadata: dict = None,
):
    """Сохраняет состояние мозга (и опционально модели мира) в .npz."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data = {
        "brain_weights": brain.weights,
        "brain_activations": brain.activations,
        "brain_trace": brain.trace,
        "brain_reward_baseline": np.array([brain.reward_baseline]),
    }

    if world_model is not None:
        data["wm_W"] = world_model.W
        data["wm_b"] = world_model.b
        data["wm_sensor_mean"] = world_model.sensor_mean
        data["wm_sensor_std"] = world_model.sensor_std

    np.savez_compressed(path, **data)
    print(f"Чекпоинт сохранён: {path}")
    print(f"  Размер: {os.path.getsize(path)} байт")
    return path


def load_checkpoint(brain, world_model=None, path: str = "checkpoints/brain_latest.npz"):
    """Загружает состояние мозга из .npz."""
    if not os.path.exists(path):
        print(f"Чекпоинт не найден: {path}")
        return False

    data = np.load(path)

    brain.weights = data["brain_weights"].copy()
    brain.activations = data["brain_activations"].copy()
    brain.trace = data["brain_trace"].copy()
    brain.reward_baseline = float(data["brain_reward_baseline"][0])

    if world_model is not None and "wm_W" in data:
        world_model.W = data["wm_W"].copy()
        world_model.b = data["wm_b"].copy()
        world_model.sensor_mean = data["wm_sensor_mean"].copy()
        world_model.sensor_std = data["wm_sensor_std"].copy()

    print(f"Чекпоинт загружен: {path}")
    return True
