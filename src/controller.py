"""
Maps classifier outputs to control signals
"""

class Controller:
    def __init__(self, smoothing=0.9):
        self.state = 0.5
        self.alpha = smoothing

    def update(self, prediction):
        """
        prediction: discrete class (0 or 1)
        """
        target = 1.0 if prediction == 1 else 0.0
        self.state = self.alpha * self.state + (1 - self.alpha) * target
        return self.state
