# Full realtime BCI pipeline

Environment: required packages are listed in requirements.txt. With differing versions of LSL, we encountered errors regarding resolving streams, so we specified pylsl==1.16.2, which works with our code.

## Running the experiment
In ``paradigm_F.py``, under ``if __name__ == '__main__':``, uncomment either the function ``simulation()`` or ``unicorn()``.

To run the paradigm with simulated EEG, you must also run ``python -m simulated_eeg.streamer`` in a separate terminal.

To run with the Unicorn, make sure the headset is paired, via the Unicorn program, and Recorder has been started. To record a new XDF file, you must also start LabRecorder (optional).

## Changing/inspecting the setup
Many variables are managed in the ``config.yaml`` file, including paradigm timing and stream names. 

## File structure

### Root
The root folder contains the experiment (``paradigm_F``), as well as requirements.txt, config.yaml, and other configuration files such as an lsl configuration file to disable a warning, and .gitignore.

### src
Src folder contains the code that the paradigm imports. 

### data
Data folder contains both the recorded XDF files, and the output of processing the XDF files into csv's where feature value and timestamp are annotated.

### examples
Examples folder contains some standalone examples, but what is relevant for the pipeline has also been implemented in the src folder. So, the examples folder can be disregarded. 

### logs
Currently, logs are being recorded for the features that are being streamed, and for the time difference after a sample has been pulled and an online classification has been made (--> pure classification time). Logs of the latency for each of the processing steps (signal acquisition, pre-processing, feature extraction, inference and feedback/control), are also logged in three separate files during each run.

## Profiling
In order to analyze the profiling data produced by our pipeline there is an additional notebook: "profiling_M.ipynb" that loads the profiling data and performs various analyses over each of the steps. 

### Results

We tested the pipeline over 963 frames, and total latency stayed well under the 100 ms threshold needed for real-time feedback. Average end-to-end delay was 41.38 ms, and 99% of all frames finished under 70.83 ms. Processing itself is extremely fast: preprocessing, feature extraction, classification, and PyGame rendering take less than 3 ms combined per frame. Almost 80% of the total delay comes from LSL sample buffering. Rendering stayed stable throughout testing, dropping only 4 frames (0.41%) at a 100 ms target stride.

