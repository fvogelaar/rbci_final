import numpy as np
import time
from pathlib import Path
import yaml
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_streams, local_clock

from src.onl_preprocessing.window import SlidingWindow
from src.features.bandpower import BandPower
from src.onl_preprocessing.filters import OnlineBandpass
from src.features.log import FeatureLogger, LatencyLogger


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"


def load_config(name="config.yaml"):
    with open(PROJECT_ROOT / name, "r") as f:
        return yaml.safe_load(f)


CONFIG = load_config()

FS = CONFIG["windows"]["fs"]
N_CHANNELS = CONFIG["windows"]["n_channels"]
WINDOW_STEP = CONFIG["windows"]["step"]
WINDOW_SIZE = CONFIG["windows"]["size"]
C3_IDX = CONFIG["windows"]["C3_IDX"]
C4_IDX = CONFIG["windows"]["C4_IDX"]


def publish_online_features(stop_event, simulation):
    streams = resolve_streams()
    eeg_stream = None

    if not simulation:
        for stream in streams:
            if stream.name() in ('UnicornRecorderLSLStream', 'MatthijsEnFrancien'):
                eeg_stream = stream
                break
    else:
        eeg_streams = [
            stream for stream in streams if stream.type().strip().lower() == "eeg"
        ]
        eeg_stream = eeg_streams[0]

    if eeg_stream is None:
        raise RuntimeError("UnicornRecorderLSLStream or MatthijsEnFrancien not found.")

    fs = eeg_stream.nominal_srate()

    if fs != FS:
        raise RuntimeError(f"Expected {FS} Hz, but the stream reports {fs} Hz.")

    if eeg_stream.channel_count() < N_CHANNELS:
        raise RuntimeError("The EEG stream has fewer than channels than expected.")

    inlet = StreamInlet(eeg_stream)

    feature_info = StreamInfo(
        name="MotorImageryFeatures",
        type="EEG_FEATURES",
        channel_count=2,
        nominal_srate=FS / WINDOW_STEP,
        channel_format="float32",
        source_id="motor-imagery-features",
    )

    feature_outlet = StreamOutlet(feature_info)

    window = SlidingWindow(size=WINDOW_SIZE, step=WINDOW_STEP)
    feature = BandPower(fs=FS, band=(8, 12))
    online_filter = OnlineBandpass(fs, 8, 30, N_CHANNELS)

    LOG_DIR.mkdir(exist_ok=True)
    time_str = time.strftime('%Y%m%d_%H%M%S')
    
    feature_logger = FeatureLogger(LOG_DIR / f"features_{time_str}.csv")
    latency_logger = LatencyLogger(LOG_DIR / f"latency_features_{time_str}.csv")

    try:
        while not stop_event.is_set():
            eeg_sample, timestamp = inlet.pull_sample(timeout=1.0)

            if eeg_sample is None:
                continue

            acquisition_latency = (local_clock() - timestamp) * 1000

            sample = np.array(eeg_sample[:N_CHANNELS], dtype=float)

            t0 = time.perf_counter()
            filtered_sample = online_filter.process(sample)
            t1 = time.perf_counter()
            
            preprocessing_latency = (t1 - t0) * 1000

            win = window.update(filtered_sample)

            if win is not None:
                feats = feature.compute(win)
                motor_power = feats[[C3_IDX, C4_IDX]]
                log_features = np.log10(motor_power + 1e-10)
                
                feature_latency = (time.perf_counter() - t1) * 1000
                latencies = (acquisition_latency, preprocessing_latency, feature_latency)

                feature_outlet.push_sample(log_features.tolist(), timestamp)
                
                feature_logger.log(log_features, timestamp)
                latency_logger.log(timestamp, latencies)

    finally:
        inlet.close_stream()
        feature_logger.close()
        latency_logger.close()


if __name__ == "__main__":
    import threading

    stop_event = threading.Event()
    publish_online_features(stop_event, simulation=True)