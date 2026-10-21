"""
Simulated closed-loop BCI demo
"""

import time
from src.feedback import VisualBar
from src.controller import Controller

fb = VisualBar()
ctrl = Controller()

predictions = [0, 1] * 50

for p in predictions:
    value = ctrl.update(p)
    fb.update(value)
    time.sleep(0.1)
