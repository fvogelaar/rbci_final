"""
Utility functions for loading feature datasets
"""

import numpy as np

def load_csv(path):
    data = np.loadtxt(path, delimiter=",")
    X = data[:, 1:-1]
    y = data[:, -1].astype(int)
    return X, y



# From assignment 4
"""
Based on loadxdfhelper
This script loads the data,
takes the EEG and marker data
calculates bandpower features
and annotates them with the condition
at that timestamp 
(or if no condition present, then the last 
known condition is annotated)
"""


import csv
import numpy as np
import pyxdf
from src.window import SlidingWindow
from src.bandpower import BandPower
from src.filters import OnlineBandpass
import pandas as pd
import os
from pathlib import Path


FS = 250
N_CHANNELS = 8
WINDOW_STEP = 25
WINDOW_SIZE = 125

# C3 C4 channels
C3_IDX, C4_IDX = 1, 3

def extract_offline_features(in_path, out_path):
    """
    Constructs the data for X from the XDF file.
    The features with their timestamps are 
    written to a CSV file.
    """
    streams, header = pyxdf.load_xdf(
        in_path,
        synchronize_clocks=False,
        dejitter_timestamps=False
    )

    # Search for the eeg stream in the XDF.
    for stream in streams:
        stream_name = stream['info']['name'][0]

        if stream_name == 'UnicornRecorderLSLStream' or stream_name == 'MatthijsEnFrancien':
            eeg_stream = stream
            # stop searching when you encounter the eeg stream
            break

    if eeg_stream is None:
        raise RuntimeError(
            "UnicornRecorderLSLStream or MatthijsEnFrancien not found in the XDF."
        )

    eeg_data = np.array(
        eeg_stream['time_series'],
        dtype=float
    )

    eeg_timestamps = np.array(eeg_stream['time_stamps'])

    fs = float(eeg_stream['info']['nominal_srate'][0])

    if fs != FS:
        raise RuntimeError(
            f"Expected {FS} Hz, but the recorded stream reports {fs} Hz."
        )

    # Unicorn presents with 9 channels.
    if eeg_data.shape[1] < N_CHANNELS:
        raise RuntimeError("The recorded stream has fewer than 8 channels.")

    # print(f"Selected EEG stream: {eeg_stream['info']['name'][0]}")
    # print(f"Recorded samples: {len(eeg_data)}")
    
    # online filtering
    eeg_data = eeg_data[:, :N_CHANNELS]     # reshape to delete the 9th channel
    online_filter = OnlineBandpass(fs, 8, 30, N_CHANNELS)
    filtered_eeg = np.zeros_like(eeg_data)
    for i, sample in enumerate(eeg_data):
        filtered_eeg[i] = online_filter.process(sample)

    window = SlidingWindow(
        size=WINDOW_SIZE,
        step=WINDOW_STEP
    )

    feature = BandPower(fs=FS, band=(8, 12))

    feature_count = 0

    with open(out_path, "w", newline="") as feature_file:
        writer = csv.writer(feature_file)

        writer.writerow([
            "timestamp",
            "C3_log_mu",
            "C4_log_mu"
        ])

        # Process recorded samples instead of pulling live samples.
        # Per - window implementation.
        for i in range(len(filtered_eeg)):
            sample = filtered_eeg[i][:N_CHANNELS]   # excludes the 9th channel
            timestamp = eeg_timestamps[i]

            win = window.update(sample)

            if win is not None:
                feats = feature.compute(win)

                motor_power = feats[[C3_IDX, C4_IDX]]

                log_features = np.log10(motor_power + 1e-10)

                writer.writerow([
                    timestamp,
                    log_features[0],
                    log_features[1]
                ])

                feature_count += 1

    # print(f"Saved {feature_count} feature windows to {out_path}")


def extract_y_labels(csv_path, in_path):
    """
    Derives the y_labels corresponding to the features
    at each timestamp. Must be run AFTER extracting 
    features.
    In the function arguments, csv_path is equal to the out_path 
    of function extract_offline_features out_path.
    """

    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / csv_path
    in_path = project_root / in_path
    #print(csv_path)
    #print(in_path)

    # If there is no previous recording, return
    if not os.path.exists(csv_path) or not os.path.exists(in_path):
        print("No previous data, or features were not calculated yet.")
        return None, None

    # LOAD XDF FILE AND SEPARATE STREAMS

    streams, header = pyxdf.load_xdf(
        in_path
    )

    marker_stream = None

    # Identify streams by type or name.
    for stream in streams:
        s_type = stream['info']['type'][0]
        s_name = stream['info']['name'][0]

        if s_type == 'Markers' or s_name == 'MotorImageryMarkers':
            marker_stream = stream

    # If no EEG or marker stream present, return.
    if marker_stream is None:
        print("marker stream missing.")
        return None, None

    marker_list = []

    for marker in marker_stream['time_series']:
        marker_list.append(marker[0])

    marker_timestamps = np.array(marker_stream['time_stamps'])

    # print(f"Loaded Marker Stream: {len(marker_list)} markers found")

    # Loading features as previously calculated by the feature pipeline
    features = pd.read_csv(csv_path)
    
    X = []
    y = []

    for i in range(len(features)):
        feature_timestamp = features['timestamp'].iloc[i]
        condition = None

        # Find the most recent imagery marker for this feature.

        # TODO: now, windows that include rest or fixation also
        #   inherit the most recent ground truth label (right/left)
        #   and it may be more accurate not to classify them

        # Here, 'condition' is the ground truth
        for j in range(len(marker_list)):

            if marker_timestamps[j] <= feature_timestamp:

                if marker_list[j] == 'IMAGERY_LEFT':
                    condition = 'LEFT'

                elif marker_list[j] == 'IMAGERY_RIGHT':
                    condition = 'RIGHT'

        # Only add features that have a condition
        if condition is not None:
            c3 = features['C3_log_mu'].iloc[i]
            c4 = features['C4_log_mu'].iloc[i]

            X.append([c3, c4])
            y.append(condition)

    # print("Number of feature rows:", len(features))

    # Test if the data is correctly composed
    if len(y) == 0:
        print("No labeled features.")
        return None, None

    if 'LEFT' not in y or 'RIGHT' not in y:
        print("Need both left and right examples to train.")
        return None, None

    # uncomment for debugging
    # print("Markers:", marker_list)
    # print("Number of labelled features:", len(y))
    # print("LEFT features:", y.count("LEFT"))
    # print("RIGHT features:", y.count("RIGHT"))

    X = np.array(X)
    y = np.array(y)

    # print(f"Training data: {len(y)} feature windows")

    return X, y
