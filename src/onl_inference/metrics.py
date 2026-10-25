"""
Real-time performance metrics
"""

import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class OnlineMetrics:
    def __init__(self):
        self.times = []

    def log_latency(self, t_start):
        self.times.append(time.perf_counter() - t_start)  # use perf_counter

    def mean_latency(self):
        if not self.times:
            return 0.0
        return sum(self.times) / len(self.times)

    def save(self):
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        path = log_dir / f"latency_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        with open(path, "w") as f:
            f.write("latency_s\n")
            for t in self.times:
                f.write(f"{t}\n")
