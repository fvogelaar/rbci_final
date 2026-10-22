import time
import random
from pylsl import StreamInfo, StreamOutlet
import winsound
import threading
from examples.train_offline import offline_trainer
from simulated_eeg.streamer import stream
from src.feature_publisher_F import publish_online_features
from src.pipeline_feedback import closed_loop_fb
from src.pipeline_online_clf import online_classification
from pathlib import Path
import yaml


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_marker_outlet(config):
    # use marker stream configuration
    marker = config["streams"]["marker"]
    info = StreamInfo(
        name=marker["name"],
        type=marker["type"],
        channel_count=1,
        nominal_srate=0,  # Irregular sampling rate for event-based triggers
        channel_format='string',
    )
    return StreamOutlet(info)


def run_motor_imagery_paradigm(outlet, config, total_trials=1):

    timing = config["timing"]
    beep = config["beep"]
    experiment = config["experiment"]
    iti_min, iti_max = timing["iti_range_seconds"]

    for trial_num in range(1, total_trials + 1):
        print(f"\nTrial {trial_num}/{total_trials}")

        # Randomly select task condition
        condition = random.choice(experiment["conditions"])
        # You can also include additional markers
        # Adjust the timings.

        # 1. FIXATION   time = 0
        print("+")
        time.sleep(timing["fixation_seconds"])

        # 2. WARNING BEEP    time = 2
        winsound.Beep(beep["frequency_hz"], beep["duration_ms"])  # in ms
        time.sleep(timing["post_beep_seconds"])

        # 3. CUE + Marker   time = 2.5
        # prediction should be made within ~1.25 s of cue
        outlet.push_sample([f"CUE_{condition}"])
        if condition == 'LEFT':
            print("<--")
        else:
            print("-->")
        time.sleep(timing["cue_seconds"])

        # 4. IMAGERY + Marker   time = 3.5
        outlet.push_sample([f"IMAGERY_{condition}"])
        time.sleep(timing["imagery_seconds"])
        # ~1.25 s after cue is point for presenting feedback

        # 5. INTER-TRIAL INTERVAL (ITI)     time = 8
        print("Rest")
        iti_seconds = random.uniform(iti_min, iti_max)
        time.sleep(iti_seconds)
        # time = 8.5 - 10.5 s


def simulation(num_trials):
    # load config
    config_path = Path(__file__).resolve().parent / "config.yaml"
    config = load_config(config_path)

    # Initialize the marker stream first
    marker_outlet = create_marker_outlet(config)

    print("LSL Marker Outlet initialized: 'MotorImageryMarkers'")

    # Linking to LabRecorder
    print("Run python -m simulated_eeg.streamer in a dedicated terminal")

    input("\n--> Press ENTER in this terminal once streamer is turned on to start the task...")

    stop_event = threading.Event()

    # Turn on feature streaming (EEG_FEATURES type stream)
    # based on the stream you want
    # Non-blocking implementation using threading
    feature_thread = threading.Thread(
        target=publish_online_features,
        args=(stop_event,),
        kwargs={"simulation": True},
        )
    feature_thread.start()

    # intiate clf thread (this produces BCI_PREDICTION type stream)
    clf_thread = threading.Thread(
            target=online_classification,
            args=(stop_event, config, cal_clf),
            kwargs={},
            )

    clf_thread.start()

    feedback_thread = threading.Thread(
                target=closed_loop_fb,
                args=(stop_event,),
                kwargs={},
                )

    feedback_thread.start()

    time.sleep(config["experiment"]["startup_delay_seconds"])
    # Start experiment.
    try:
        run_motor_imagery_paradigm(
            marker_outlet,
            config,
            total_trials=num_trials,
        )

    # Final block runs also when we get an error before
    # To make sure extraction stops
    finally:
        # Stop feature extraction when the experiment finishes
        # So completing the total_trials is the stop_event for
        # feature extraction and for the optional simulation stream
        stop_event.set()
        feature_thread.join()


def unicorn(num_trials):
    # load config
    config_path = Path(__file__).resolve().parent / "config.yaml"
    config = load_config(config_path)

    # Initialize the marker stream first
    marker_outlet = create_marker_outlet(config)

    print("LSL Marker Outlet initialized: 'MotorImageryMarkers'")

    # Linking to LabRecorder
    print("0. Start Unicorn Recorder and run the paradigm code.")
    print("\n1. Open LSL LabRecorder.")
    print("2. Click 'Update' and ensure both 'Unicorn' and 'MotorImageryMarkers' are visible.")
    print("3. Select both streams and click 'Start' to begin recording.")

    input("\n--> Press ENTER in this terminal once LabRecorder is recording to start the task...")

    stop_event = threading.Event()

    # Turn on feature streaming (EEG_FEATURES type stream)
    # based on the stream you want
    # Non-blocking implementation using threading
    feature_thread = threading.Thread(
        target=publish_online_features,
        args=(stop_event,),
        kwargs={"simulation": False},
        )
    feature_thread.start()

    # intiate clf thread (this produces BCI_PREDICTION type stream)
    clf_thread = threading.Thread(
            target=online_classification,
            args=(stop_event, config, cal_clf),
            kwargs={},
            )

    clf_thread.start()

    feedback_thread = threading.Thread(
                target=closed_loop_fb,
                args=(stop_event,),
                kwargs={},
                )

    feedback_thread.start()

    time.sleep(config["experiment"]["startup_delay_seconds"])
    # Start experiment.
    try:
        run_motor_imagery_paradigm(
            marker_outlet,
            config,
            total_trials=num_trials,
        )

    # Final block runs also when we get an error before
    # To make sure extraction stops
    finally:
        # Stop feature extraction when the experiment finishes
        # So completing the total_trials is the stop_event for
        # feature extraction and for the optional simulation stream
        stop_event.set()
        feature_thread.join()


if __name__ == '__main__':
    cal_clf = offline_trainer()
    simulation(num_trials=2)
    # unicorn(num_trials=2)
