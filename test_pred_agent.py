"""
Тест PredictiveBrain в реальной среде.
Агент пытается удержать CPU и GPU в зоне (0.45 - 0.55).
"""

import time
import numpy as np
from predictive_brain import PredictiveBrain
from body_sensor import BodySensor
from body_actuator import BodyActuator
from energy import Energy

def main():
    sensor = BodySensor()
    actuator = BodyActuator()
    energy = Energy()
    
    # 4 входа: cpu_load, gpu_load, target_cpu, target_gpu
    # 2 выхода: cpu_intensity, gpu_intensity
    brain = PredictiveBrain(n_sensors=2, n_actions=2, lr=0.05, lr_action=0.5)
    
    actuator.start()
    time.sleep(1.0) # Прогрев
    
    # Цели (центр зоны)
    target_cpu = 0.5
    target_gpu = 0.5
    target = np.array([target_cpu, target_gpu])
    
    print("Шаг | Жизнь | Энергия | CPU | GPU | Ошибка предсказания")
    print("-" * 65)
    
    step = 0
    try:
        while step < 500:
            # 1. Сенсоры
            _, state = sensor.read()
            cpu = state.get("cpu_load", 0.0)
            gpu = state.get("gpu_load", 0.0)
            
            # Вход для мозга: только реальные сенсоры
            inputs = np.array([cpu, gpu])
            
            # 2. Мозг думает (Active Inference)
            action = brain.act(inputs, target)
            # Исследование: 5% случайного шума, чтобы мозг не залипал
            action += np.random.normal(0.0, 0.02, size=2)
            action = np.clip(action, 0.0, 1.0)
            
            # 3. Действие
            actuator.set_intensity(action[0], action[1])
            time.sleep(0.05)
            
            # 4. Новое состояние
            _, next_state = sensor.read()
            next_cpu = next_state.get("cpu_load", 0.0)
            next_gpu = next_state.get("gpu_load", 0.0)
            
            # 5. Мозг учится предсказывать
            next_inputs = np.array([next_cpu, next_gpu])
            err = brain.learn(inputs, action, next_inputs)
            
            # 6. Энергия
            energy.update(next_state)
            
            if step % 50 == 0:
                print(f"{step:3d} | {energy.energy:.3f} | {cpu:.2f} | {gpu:.2f} | {err:.4f}")
            
            if not energy.is_alive():
                print(f"*** Смерть на такте {step} ***")
                break
                
            step += 1
            
    finally:
        actuator.stop()
        print(f"Прожито тактов: {step}")

if __name__ == "__main__":
    main()
