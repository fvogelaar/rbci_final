"""
Closed-loop real-time feedback pipeline
"""

import time
from pathlib import Path
from pylsl import StreamInlet, resolve_stream, local_clock
from src.feedback_control.feedback import VisualBar
from src.feedback_control.controller import Controller, reset_event
from src.features.log import FeedbackLatencyLogger
import pygame

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"


def closed_loop_fb(stop_event):
    streams = resolve_stream("type", "BCI_PREDICTION")
    inlet = StreamInlet(streams[0])

    feedback = VisualBar()
    controller = Controller(smoothing=0.9)
    clock = pygame.time.Clock()
    
    latest_y = None
    latest_timestamp = None

    LOG_DIR.mkdir(exist_ok=True)
    time_str = time.strftime('%Y%m%d_%H%M%S')
    latency_logger = FeedbackLatencyLogger(LOG_DIR / f"latency_feedback_{time_str}.csv")

    try:
        while not stop_event.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    stop_event.set()

            if stop_event.is_set():
                break

            if reset_event.is_set():
                controller.reset()
                latest_y = None
                latest_timestamp = None
                reset_event.clear()
                while inlet.pull_sample(timeout=0.0)[0] is not None:
                    pass

            while True:
                pred, timestamp = inlet.pull_sample(timeout=0.0)

                if pred is None:
                    break

                latest_y = pred[0]
                latest_timestamp = timestamp

            t0 = time.perf_counter()

            if latest_y is not None:
                value = controller.update(latest_y)
            else:
                value = controller.state

            feedback.update(value)
            
            if latest_timestamp is not None:
                fb_latency = (time.perf_counter() - t0) * 1000
                total_latency = (local_clock() - latest_timestamp) * 1000
                latency_logger.log(latest_timestamp, fb_latency, total_latency)

            clock.tick(20)

    finally:
        latency_logger.close()
        inlet.close_stream()
        pygame.quit()


if __name__ == "__main__":
    closed_loop_fb()