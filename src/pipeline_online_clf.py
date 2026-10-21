"""
Real-time classification pipeline
"""

import time
from pylsl import StreamInlet, resolve_stream, StreamOutlet, StreamInfo
from src.classifier import OnlineLDA
from src.metrics import OnlineMetrics
from examples.train_offline import offline_trainer

def online_classification(stop_event):
    streams = resolve_stream("type", "EEG_FEATURES")
    inlet = StreamInlet(streams[0])

    # create outlet for BCI_PREDICTION probabilities
    prediction_info = StreamInfo(
        name="MotorImageryPredictions",
        type="BCI_PREDICTION",
        channel_count=1,
        nominal_srate=0,
        channel_format="float32",
        source_id="motor-imagery-predictions",
    )

    prediction_outlet = StreamOutlet(prediction_info)

    cal_model = offline_trainer()     # this runs the offline training with calibration file
    metrics = OnlineMetrics()


    try:
        while not stop_event.is_set():
            t0 = time.time()
            feature, _ = inlet.pull_sample()

            y = cal_model.predict(feature)
            #probabilities = cal_model.pred_probability(feature)
            prediction_outlet.push_sample([y])
            
            if y is not None:
                metrics.log_latency(t0)
                print("Prediction:", y,
                    "Mean latency:", metrics.mean_latency())
    
    finally:
        inlet.close_stream()

if __name__ == "__main__":
    online_classification()