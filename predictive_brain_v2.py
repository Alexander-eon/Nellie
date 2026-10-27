import numpy as np

class PredictiveBrainV2:
    """
    Улучшенная версия PredictiveBrain.
    Разделяет предсказание (W_s) и выбор действия (W_a).
    act() возвращает действие (n_actions), а не предсказание состояния.
    """
    def __init__(self, n_sensors, n_actions, lr=0.01, lr_action=0.1):
        self.n_s = n_sensors
        self.n_a = n_actions
        self.lr = lr
        self.lr_action = lr_action
        
        # Модель мира: предсказывает следующее состояние
        self.W_s = np.random.randn(n_sensors, n_sensors) * 0.01
        
        # Модель действия: как действие влияет на следующее состояние
        self.W_a = np.zeros((n_sensors, n_actions))
        
        self.last_action = np.ones(n_actions) * 0.5

    def predict(self, sensors, action):
        """Предсказать следующее состояние."""
        return self.W_s @ sensors + self.W_a @ action

    def learn(self, sensors, action, next_sensors):
        """Обновить модель на основе реального следующего состояния."""
        pred = self.predict(sensors, action)
        error = next_sensors - pred
        
        # Обновляем модель мира
        self.W_s += self.lr * np.outer(error, sensors)
        # Обновляем модель действия
        self.W_a += self.lr * np.outer(error, action)
        
        return float(np.mean(np.abs(error)))

    def act(self, sensors, target_state):
        """
        Выбрать действие, чтобы достичь target_state.
        Возвращает действие (n_actions).
        """
        action = self.last_action.copy()
        
        # Градиентный спуск по действию
        for _ in range(20):
            pred = self.predict(sensors, action)
            error = pred - target_state
            # Градиент по действию: W_a.T @ error
            action -= self.lr_action * (self.W_a.T @ error)
            action = np.clip(action, 0.0, 1.0)

        self.last_action = action.copy()
        return action
