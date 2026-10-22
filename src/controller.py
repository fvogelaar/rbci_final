"""
Maps classifier outputs to control signals
"""


class Controller:
    def __init__(self, smoothing=0.9):
        self.state = 0.5
        self.alpha = smoothing

    def update(self, prediction):
        """
        prediction: probability output from clf
        predict_proba gives e.g. [0.8, 0.2] (L, R)
        and then the BCI_FEATURES stream gives the first
        element of that (the left probability)

        """
        # if P(Left) = 1.00, target = -1
        # if P(Left) = 0.00, target = +1
        print(prediction)
        target = 1 - 2 * prediction
        
        self.state = self.alpha * self.state + (1 - self.alpha) * target
        
        return self.state
