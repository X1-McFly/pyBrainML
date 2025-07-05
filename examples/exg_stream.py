"""
EXG Streaming and Real-Time Plotting Example

Author: Martin McCorkle
Date: 2025-07-03
Description:
    Demonstrates how to stream EXG data from a BrainFlow-compatible board using
    the pybrainml library, save raw data to an NDJSON file, and plot all EXG
    channels in real time.

Dependencies:
    - pybrainml>=0.3.2
    - matplotlib
    - pandas
    - brainflow
    - yaspin
"""

import time
from typing import Deque, List

import matplotlib.pyplot as plt

import pybrainml as bml
from pybrainml import ElectrodeType, Boards

def main():

    #Experiment setup
    port = "COM8"
    data_dir = "data"
    window_length = 200

    exp = bml.create_experiment()
    exp.user_setup("John Doe", 35, "F")
    exp.hardware_setup(ElectrodeType.HYBRID, Boards.OpenBCI_Ganglion)


    # Connect to board and prepare streaming session
    board_fd = bml.connect_board(port, Boards.OpenBCI_Ganglion)
    session = bml.exg_stream(board_fd, length=window_length)


    # Prepare real-time plot
    plt.ion()
    fig, ax = plt.subplots()
    num_ch = len(session.eeg_channels())
    lines = [ax.plot([], [], label=f"Chan {i+1}")[0] for i in range(num_ch)]
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("EXG Value")
    ax.set_xlim(0, window_length)
    ax.legend().set_visible(False)

    # Start experiment
    session.start()

    try:
        while True:
            time.sleep(0.01)
            buf: Deque[List[float | str]] = session.get_buffer()
            if not buf:
                continue
            x = list(range(len(buf)))
            channel_traces: List[List[float]] = [[float(sample[i+1]) for sample in buf] for i in range(num_ch)]
            for line, vals in zip(lines, channel_traces):
                line.set_data(x, vals)
            all_vals = [v for vals in channel_traces for v in vals]
            if all_vals:
                ax.set_ylim(min(all_vals) * 1.1, max(all_vals) * 1.1)
            
            fig.canvas.draw()
            fig.canvas.flush_events()

            if not session.is_running():
                break

    except KeyboardInterrupt:
        pass

    finally:
        print("Stopping session...")
        session.stop()
        bml.export_experiment(session, exp, data_dir)
        plt.ioff()
        plt.close()
    return

if __name__ == "__main__":
    main()