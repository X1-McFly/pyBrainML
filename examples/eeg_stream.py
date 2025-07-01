"""
EEG Streaming and Real-Time Plotting Example

Author: Martin McCorkle
Date: 2025-06-30
Description:
    Demonstrates how to stream EEG data from a BrainFlow-compatible board using
    the pybrainml library, save raw data to an NDJSON file, and plot all EEG
    channels in real time.

Dependencies:
    - pybrainml>=0.3
    - matplotlib
    - pandas
    - brainflow
    - yaspin
"""

import time
import json
from datetime import datetime
from typing import Deque, List

import matplotlib.pyplot as plt

import pybrainml as bml
from pybrainml import ElectrodeType, Boards, Frame
import os


def main():
    port = "COM8"
    window_length = 200

    # Build experiment metadata
    exp = bml.create_experiment()
    exp.metadata.subject_info.setup("John Doe", 35, "F")
    exp.metadata.hardware_info.setup(ElectrodeType.HYBRID, Boards.OpenBCI_Ganglion)

    # Start streaming in background
    temp_f = "eeg.ndjson"
    session = bml.start_eeg_stream(port=port, save_to=temp_f, length=window_length)
    print(f"Streaming started on {port}...")

    # Determine number of EEG channels
    _, sampling_rate, eeg_chs = bml.init_board(port)
    num_ch = len(eeg_chs)

    # Prepare real-time plot
    plt.ion()
    fig, ax = plt.subplots()
    lines = [ax.plot([], [], label=f"Chan {i+1}")[0] for i in range(num_ch)]
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("EEG Value")
    ax.set_xlim(0, window_length)
    ax.legend().set_visible(False)

    try:
        while True:
            time.sleep(0.01)
            buf: Deque[List[float | str]] = session.get_buffer()
            if not buf:
                continue

            x = list(range(len(buf)))
            # sample format: [timestamp_str, ch1, ch2, ...]
            channel_traces: List[List[float]] = [
                [float(sample[i+1]) for sample in buf]
                for i in range(num_ch)
            ]

            for line, vals in zip(lines, channel_traces):
                line.set_data(x, vals)

            all_vals = [v for vals in channel_traces for v in vals]
            if all_vals:
                ax.set_ylim(min(all_vals) * 1.1, max(all_vals) * 1.1)

            fig.canvas.draw()
            fig.canvas.flush_events()

    except KeyboardInterrupt:
        print("Stopping session...")

    finally:
        session.stop()
        
        filename = bml.get_unique_file("test.json")
        # Persist experiment metadata
        processed_frame = bml.post_process(temp_f)
        # time.sleep(1)
        if os.path.exists(temp_f):
            os.remove(temp_f)
        if processed_frame is not None:
            print(f"Saving processed frame to {filename}...")
            exp.frames.append(processed_frame)
            with open(filename, "w") as f:
                json.dump(exp.to_dict(), f, indent=4)
        else:
            print("No processed frame to save.")
        
        plt.ioff()
        plt.close()
        print(f"Done.")
    return

if __name__ == "__main__":
    main()
