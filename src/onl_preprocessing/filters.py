"""
Causal EEG filters for real-time use
"""

import numpy as np
from scipy.signal import butter, lfilter


class OnlineBandpass:
    """
    Stateful causal band-pass filter
    """
    def __init__(self, fs, low, high, n_channels, order=4):
        self.b, self.a = butter(
            order, [low, high],
            btype="band", fs=fs
        )
        self.zi = np.zeros((max(len(self.a), len(self.b)) - 1, n_channels))

    def process(self, sample):
        """
        Process one multichannel EEG sample
        """
        y, self.zi = lfilter(
            self.b, self.a,
            [sample], axis=0, zi=self.zi
        )
        return y[0]

    def process_chunk(self, samples):
        """
        Process multiple samples at once (chunk).

        Shape:
        (number_of_samples, number_of_channels)
        Essentially instead of 1 sample we pass a
        sample matrix to lfilter.
        """
        y, self.zi = lfilter(
            self.b, self.a,
            samples, axis=0,
            zi=self.zi
        )
        return y


class Stateless:
    """
    For question 1:
    Stateless causal band-pass filter
    """
    def __init__(self, fs, low, high, n_channels, order=4):
        self.b, self.a = butter(
            order, [low, high],
            btype="band", fs=fs
        )
        self.zi = np.zeros((max(len(self.a), len(self.b)) - 1, n_channels))
        self.n_channels = n_channels

    def process(self, sample):
        """
        Process one multichannel EEG sample
        """
        zi = np.zeros(
            (max(len(self.a), len(self.b)) - 1, self.n_channels)
        )

        y, _ = lfilter(
            self.b, self.a,
            [sample], axis=0, zi=zi)
        # note that zi is not self.zi but np.zeros
        return y[0]
