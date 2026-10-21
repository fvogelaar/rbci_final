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

FIXATION_SECONDS = 2.0
CUE_SECONDS = 1.0
POST_BEEP_SECONDS = 0.5
IMAGERY_SECONDS = 4.5


def create_marker_outlet():
    info = StreamInfo(
        name='MotorImageryMarkers',
        type='Markers',
        channel_count=1,
        nominal_srate=0,  # Irregular sampling rate for event-based triggers
        channel_format='string',
    )
    return StreamOutlet(info)


def run_motor_imagery_paradigm(outlet, total_trials=1):

    conditions = ['LEFT', 'RIGHT']

    for trial_num in range(1, total_trials + 1):
        print(f"\nTrial {trial_num}/{total_trials}")

        # Randomly select task condition
        condition = random.choice(conditions)
        # You can also include additional markers
        # Adjust the timings.

        # 1. FIXATION   time = 0
        print("+")
        time.sleep(FIXATION_SECONDS)

        # 2. WARNING BEEP    time = 2
        winsound.Beep(frequency=1000, duration=500)  # in ms
        time.sleep(POST_BEEP_SECONDS)

        # 3. CUE + Marker   time = 2.5
        # prediction should be made within ~1.25 s of cue
        outlet.push_sample([f"CUE_{condition}"])
        if condition == 'LEFT':
            print("<--")
        else:
            print("-->")
        time.sleep(CUE_SECONDS)

        # 4. IMAGERY + Marker   time = 3.5
        outlet.push_sample([f"IMAGERY_{condition}"])
        time.sleep(IMAGERY_SECONDS)
        # ~1.25 s after cue is point for presenting feedback

        # 5. INTER-TRIAL INTERVAL (ITI)     time = 8
        print("Rest")
        iti_seconds = random.uniform(0.5, 2.5)
        time.sleep(iti_seconds)
        # time = 8.5 - 10.5 s


if __name__ == '__main__':
    # Initialize the marker stream first
    marker_outlet = create_marker_outlet()

    print("LSL Marker Outlet initialized: 'MotorImageryMarkers'")

    # Linking to LabRecorder
    print("0. Start Unicorn Recorder and run the paradigm code.")
    print("\n1. Open LSL LabRecorder.")
    print("2. Click 'Update' and ensure both 'Unicorn' and 'MotorImageryMarkers' are visible.")
    print("3. Select both streams and click 'Start' to begin recording.")

    input("\n--> Press ENTER in this terminal once LabRecorder is recording to start the task...")

    stop_event = threading.Event()

    # Turn on streamer if simulating eeg
    #stream(stop_event)

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
            args=(stop_event,),
            kwargs={},
            )
    
    clf_thread.start()

    feedback_thread = threading.Thread(
                target=closed_loop_fb,
                args=(stop_event,),
                kwargs={},
                )
        
    feedback_thread.start()

    time.sleep(7)
    # Start experiment.
    try:
        run_motor_imagery_paradigm(
            marker_outlet,
            total_trials=5,
        )

    # Final block runs also when we get an error before
    # To make sure extraction stops
    finally:
        # Stop feature extraction when the experiment finishes
        # So completing the total_trials is the stop_event for
        # feature extraction and for the optional simulation stream
        stop_event.set()
        feature_thread.join()
        
