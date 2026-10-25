"""
Simple feature logger
"""

import csv
import time

class FeatureLogger:
    def __init__(self, filename):
        self.file = open(filename, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["timestamp", "C3_log_mu", "C4_log_mu"])

    def log(self, features, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
            
        self.writer.writerow([timestamp] + features.tolist())
        self.file.flush()
