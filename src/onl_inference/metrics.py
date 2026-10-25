"""
Real-time performance metrics
"""
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class OnlineMetrics:
    def __init__(self, prefix="latency_ml"):
        self.prefix = prefix
        self.records = []

    def log_latency(self, timestamp, t_start):
        latency_ms = (time.perf_counter() - t_start) * 1000
        self.records.append((timestamp, latency_ms))

    def mean_latency(self):
        if not self.records:
            return 0.0
        return sum(r[1] for r in self.records) / len(self.records)

    def save(self):
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        path = log_dir / f"{self.prefix}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(path, "w") as f:
            f.write("Timestamp,Latency_ms\n")
            for ts, lat in self.records:
                f.write(f"{ts},{lat}\n")

