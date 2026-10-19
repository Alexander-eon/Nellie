import numpy as np

class PredictiveBrain:
    def __init__(self, n_sensors: int, n_actions: int, lr: float = 0.01, lr_action: float = 0.1):
        self.n_s = n_sensors
        self.n_a = n_actions
        self.lr = lr
        self.lr_action = lr_action

        self.W_s = np.random.randn(self.n_s, self.n_s) * 0.01
        self.W_a = np.zeros((self.n_s, self.n_a))
        self.last_action = np.ones(self.n_a) * 0.5

    def predict(self, sensors: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self.W_s @ sensors + self.W_a @ action

    def learn(self, sensors: np.ndarray, action: np.ndarray, next_sensors: np.ndarray) -> float:
        pred = self.predict(sensors, action)
        error = next_sensors - pred
        
        self.W_s += self.lr * np.outer(error, sensors)
        self.W_a += self.lr * np.outer(error, action)
        
        return float(np.mean(np.abs(error)))

    def act(self, sensors: np.ndarray, target: np.ndarray) -> np.ndarray:
        action = self.last_action.copy()
        
        for _ in range(20):
            pred = self.predict(sensors, action)
            error = pred - target
            
            # Ошибку по сообщению (индексы 1,2,3) обнуляем для градиента (агент не управляет чужими сообщениями)
            if len(error) > 1:
                error[1:] = 0.0 
            
            action -= self.lr_action * (self.W_a.T @ error)
            action = np.clip(action, 0.0, 1.0)

        # Агент напрямую устанавливает свои сообщения (индексы 1,2,3) равными целевым (target[1:])
        action[1:] = target[1:]

        self.last_action = action.copy()
        return action
