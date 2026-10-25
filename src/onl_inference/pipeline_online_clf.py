"""
Real-time classification pipeline
"""

import time
from pylsl import StreamInlet, resolve_stream, StreamOutlet, StreamInfo
from src.onl_inference.metrics import OnlineMetrics
from src.offl_training.train_offline import offline_trainer


def online_classification(stop_event, config, cal_clf):
    stream_settings = config["streams"]
    streams = resolve_stream("type", stream_settings["features"]["type"])
    inlet = StreamInlet(streams[0])

    # create outlet for BCI_PREDICTION probabilities
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
    # this runs the offline training with calibration file
    # cal_model = offline_trainer()
    metrics = OnlineMetrics()

    try:
        while not stop_event.is_set():
            t0 = time.time()
            feature, _ = inlet.pull_sample()

            # y = cal_model.predict(feature)
            probs = cal_model.pred_probability(feature)
            #print(probs)
            # feed P(L) into stream
            y = probs[0]
            #print(y)
            prediction_outlet.push_sample([y])

            if y is not None:
                metrics.log_latency(t0)
                # print("Prediction:", y,
                    # "Mean latency:", metrics.mean_latency())

    finally:
        inlet.close_stream()


if __name__ == "__main__":
    online_classification()
