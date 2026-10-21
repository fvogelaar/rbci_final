"""
Calculate features from realtime Unicorn EEG and publish them over LSL.
"""

import numpy as np

from pylsl import (
    StreamInfo,
    StreamOutlet,
    StreamInlet,
    resolve_streams,
)

from src.window import SlidingWindow
from src.bandpower import BandPower
from src.filters import OnlineBandpass


FS = 250
N_CHANNELS = 8
WINDOW_STEP = 25
WINDOW_SIZE = 125

C3_IDX, C4_IDX = 1, 3


def publish_online_features(stop_event, simulation):

    # resolve live stream
    streams = resolve_streams()

    eeg_stream = None

    if simulation == False: 
        # for all connected streams, find the first one called
        # UnicornRecorderLSLStream
        for stream in streams:
            if stream.name() == "UnicornRecorderLSLStream" or stream.name() == "MatthijsEnFrancien":
                eeg_stream = stream
                break
    else:
        # grab the first EEG type stream
        eeg_streams = [
        stream for stream in streams if stream.type().strip().lower() == "eeg"
        ]
        eeg_stream = eeg_streams[0]

    if eeg_stream is None:
        raise RuntimeError(
            "UnicornRecorderLSLStream or MatthijsEnFrancien not found."
        )

    fs = eeg_stream.nominal_srate()

    if fs != FS:
        raise RuntimeError(
            f"Expected {FS} Hz, but the stream reports {fs} Hz."
        )

    if eeg_stream.channel_count() < N_CHANNELS:
        raise RuntimeError(
            "The EEG stream has fewer than channels than expected."
        )

    inlet = StreamInlet(eeg_stream)     # INLET = Unicorn EEG

    # Create the feature stream. 
    feature_info = StreamInfo(
        name="MotorImageryFeatures",
        type="EEG_FEATURES",
        channel_count=2,
        nominal_srate=FS / WINDOW_STEP,
        channel_format="float32",
        source_id="motor-imagery-features",
    )

    feature_outlet = StreamOutlet(feature_info) # OUTLET = features

    # Uncomment for debugging
    # print(f"Selected EEG stream: {eeg_stream.name()}")
    # print(f"Publishing {feature_info.channel_count()} channels"
          # f" to stream of type {feature_info.type()}")

    # initiate windowing
    window = SlidingWindow(
        size=WINDOW_SIZE,
        step=WINDOW_STEP,
    )

    # set bandpower settings, here 8-12 Hz
    feature = BandPower(fs=FS, band=(8, 12))

    # initiate online filter
    online_filter = OnlineBandpass(fs, 8, 30, N_CHANNELS)

    try:
        while not stop_event.is_set():

            # Read one live sample instead of one XDF row
            eeg_sample, timestamp = inlet.pull_sample(timeout=1.0)

            if eeg_sample is None:
                continue

            # Exclude the 9th channel
            sample = np.array(
                eeg_sample[:N_CHANNELS],
                dtype=float,
            )
            # apply online filter
            filtered_sample = online_filter.process(sample)
        
            # Populate the window with the sample. 
            win = window.update(filtered_sample)

            if win is not None:
                feats = feature.compute(win)

                motor_power = feats[[C3_IDX, C4_IDX]]

                log_features = np.log10(
                    motor_power + 1e-10
                )

                # Publish per ready window
                feature_outlet.push_sample(
                    log_features.tolist(),
                    timestamp,
                )
                # uncomment for debugging
                # print("Published features:", log_features, flush=True)
    # If anything errors or the recording is finished, 
    # close the inlet to stop receiving EEG.
    finally:
        inlet.close_stream()

if __name__ == "__main__":
    import threading

    stop_event = threading.Event()
    publish_online_features(stop_event, simulation=True)