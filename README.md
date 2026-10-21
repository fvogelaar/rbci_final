# bci-online-classification

This repository focuses on deploying machine learning models
in real-time EEG pipelines.

Key topics:
- Train offline, infer online
- Latency-aware prediction
- Simple models > complex models in real time

Models we will use:
- Linear Discriminant Analysis (LDA)

# bci-feedback-loop

This repository implements real-time feedback for BCI systems.
Feedback timing and stability are critical for learning and control.

Feedback types that we will cover here:
- Visual bar feedback
- Continuous control signals

# Simulated EEG stream option

We have laid out this repository so that it works both with real eeg streams from the unicorn unit, but also with the simulated eeg stream produced by streamer.py. To use simulated EEG signals, the simulation argument in line 95 of paradigm_F.py needs to be set to true and simulated_eeg.streamer needs to be run in a seperate terminal window. 