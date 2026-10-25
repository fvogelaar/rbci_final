# Full realtime BCI pipeline

Environment: required packages are listed in requirements.txt. With differing versions of LSL, we encountered errors regarding resolving streams, so we specified pylsl==1.16.2, which works with our code.

## Running the experiment
In ``paradigm_F.py``, under ``if __name__ == '__main__':``, uncomment either the function ``simulation()`` or ``unicorn()``.

To run the paradigm with simulated EEG, you must also run ``python -m simulated_eeg.streamer`` in a separate terminal.

To run with the Unicorn, make sure the headset is paired, via the Unicorn program, and Recorder has been started. To record a new XDF file, you must also start LabRecorder (optional).

## Changing/inspecting the setup
Many variables are managed in the ``config.yaml`` file, including paradigm timing and stream names. 