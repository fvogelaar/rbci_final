from src.feedback_control.feedback import VisualBar
import time
bar = VisualBar()
for v in (0.8, 0.0, -0.8):
    bar.update(v)
    time.sleep(2)