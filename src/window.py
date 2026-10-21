"""
Sliding window for streaming EEG samples
"""

import numpy as np

class SlidingWindow:
    def __init__(self, size, step):
        self.size = size
        self.step = step
        self.buffer = []
        self.counter = 0

    def update(self, sample):
        self.buffer.append(sample)
        self.counter += 1

        if len(self.buffer) > self.size:
            self.buffer.pop(0)

        if len(self.buffer) == self.size and self.counter % self.step == 0:
            return np.array(self.buffer)

        return None
