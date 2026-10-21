"""
Simulated EEG stream using Lab Streaming Layer (LSL)
Copy of file given in previous assignments.
"""

import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet

FS = 250
N_CHANNELS = 8


def stream():
  
    info = StreamInfo(
        name="SimulatedEEG",
        type="EEG",
        channel_count=N_CHANNELS,
        nominal_srate=FS,
        channel_format="float32",
        source_id="sim_eeg_001"
    )

    outlet = StreamOutlet(info)
    print("Streaming simulated EEG...")

    while True:
        sample = np.random.randn(N_CHANNELS).astype(np.float32)
        outlet.push_sample(sample)
        time.sleep(1.0 / FS)


if __name__ == "__main__":
    stream()
