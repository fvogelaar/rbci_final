"""
Standalone feedback test

Not our full implementation.
"""

import time
from src.feedback_control.feedback import VisualBar

fb = VisualBar()
v = 0.0
direction = 1

while True:
    v += 0.01 * direction
    if v >= 1.0 or v <= 0.0:
        direction *= -1
    fb.update(v)
    time.sleep(0.02)
