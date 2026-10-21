"""
Band power feature extraction
"""

import numpy as np
from scipy.signal import welch

class BandPower:
    def __init__(self, fs, band):
        self.fs = fs
        self.band = band

    def compute(self, window):
        """
        window shape: (samples, channels)
        """
        freqs, psd = welch(window, fs=self.fs, axis=0, nperseg=min(256, window.shape[0]))
        idx = (freqs >= self.band[0]) & (freqs <= self.band[1])
        return psd[idx].mean(axis=0)