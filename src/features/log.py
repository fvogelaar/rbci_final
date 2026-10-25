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

    def close(self):
        self.file.close()


class LatencyLogger:
    def __init__(self, filename):
        self.file = open(filename, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["Timestamp", "Acquisition_ms", "Preprocessing_ms", "Feature_ms"])

    def log(self, timestamp, latencies):
        self.writer.writerow([timestamp] + list(latencies))
        self.file.flush()

    def close(self):
        self.file.close()

import csv

class FeedbackLatencyLogger:
    def __init__(self, filename):
        self.file = open(filename, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["Timestamp", "Feedback_Update_ms", "Total_End_to_End_ms"])

    def log(self, timestamp, fb_latency, total_latency):
        self.writer.writerow([timestamp, fb_latency, total_latency])
        self.file.flush()

    def close(self):
        self.file.close()