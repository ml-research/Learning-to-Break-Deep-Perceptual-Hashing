import numpy as np

class EarlyStopper:
    def __init__(self, window, min_diff=0.005):
        self.window = window
        self.best_value = np.inf
        self.current_count = 0
        self.min_diff = min_diff

    def stop_early(self, value):
        if self.best_value <= (value + self.min_diff) and self.current_count >= self.window:
            self.current_count = 0
            return True

        if value < self.best_value and (self.best_value - value) >= self.min_diff:
            self.current_count = 0
            self.best_value = value

        self.current_count += 1

        return False