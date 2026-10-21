"""
Real-time performance metrics
"""

import time

class OnlineMetrics:
    def __init__(self):
        self.times = []

    def log_latency(self, t_start):
        self.times.append(time.time() - t_start)

    def mean_latency(self):
        if not self.times:
            return 0.0
        return sum(self.times) / len(self.times)
