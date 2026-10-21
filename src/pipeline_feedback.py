"""
Closed-loop real-time feedback pipeline
"""

from pylsl import StreamInlet, resolve_stream
from src.feedback import VisualBar
from src.controller import Controller
import pygame

def closed_loop_fb(stop_event):
    streams = resolve_stream("type", "BCI_PREDICTION")
    inlet = StreamInlet(streams[0])

    feedback = VisualBar()
    controller = Controller(smoothing=0.5)     #high smoothing value to prevent bar draining quickly
    clock = pygame.time.Clock()
    latest_y = None

    try:
        while not stop_event.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    stop_event.set()

            if stop_event.is_set():
                break

            while True:
                pred, _ = inlet.pull_sample(timeout=0.0)
                
                if pred is None:
                    break

                latest_y = int(pred[0])
                
            if latest_y is not None:
                value = controller.update(latest_y)
            else:
                value = controller.state
                
            feedback.update(value)
            # 60 frames per second
            
            clock.tick(20)

            
    finally:
        inlet.close_stream()
        pygame.quit()

if __name__ == "__main__":
    closed_loop_fb()