"""
Real-time classification pipeline
"""

import time
from pylsl import StreamInlet, resolve_stream, StreamOutlet, StreamInfo
from src.onl_inference.metrics import OnlineMetrics


def online_classification(stop_event, config, cal_clf):
    stream_settings = config["streams"]
    streams = resolve_stream("type", stream_settings["features"]["type"])
    inlet = StreamInlet(streams[0])

    prediction_info = StreamInfo(
        name=stream_settings["predictions"]["name"],
        type=stream_settings["predictions"]["type"],
        channel_count=1,
        nominal_srate=0,
        channel_format="float32",
        source_id=stream_settings["predictions"]["source_id"],
    )

    prediction_outlet = StreamOutlet(prediction_info)
    cal_model = cal_clf
    metrics = OnlineMetrics(prefix="latency_ml")

    try:
        while not stop_event.is_set():
            # Catch the LSL timestamp
            feature, timestamp = inlet.pull_sample()
            
            if feature is None:
                continue
                
            t0 = time.perf_counter()
            probs = cal_model.pred_probability(feature)
            # print(probs)
            y = probs[0]
            
            # Push original timestamp forward to the feedback loop
            prediction_outlet.push_sample([y], timestamp)

            if y is not None:
                metrics.log_latency(timestamp, t0)
                
    finally:
        metrics.save()
        inlet.close_stream()


if __name__ == "__main__":
    online_classification()
