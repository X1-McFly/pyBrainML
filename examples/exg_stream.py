"""
EXG Streaming and Real-Time Plotting Example (Basic File Storage)

Author: Martin McCorkle
Date: 2025-07-03
Description:
    Demonstrates basic EXG data streaming from a BrainFlow-compatible board using
    the pybrainml library. Saves raw data to NDJSON files and plots all EXG
    channels in real time.

Dependencies:
    - pybrainml>=0.3.2
    - matplotlib
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
    port = "COM8"  # For real board
    # For synthetic board testing, comment out the port line above and use:
    # port = None
    data_dir = "data"
    window_length = 200
    exp = bml.create_experiment()
    exp.user_setup(None, 35, "F")
    
    # Choose board type:
    # exp.hardware_setup(ElectrodeType.DRY, Boards.OpenBCI_Ganglion)
    exp.hardware_setup(ElectrodeType.DRY, Boards.Synthetic)  # Synthetic board for testing

    # Connect to board and prepare streaming session
    try:
        board_fd = bml.connect_board(exp, port) if port else bml.connect_board(exp)
    except ConnectionError as e:
        print(f"Failed to connect to board: {e}")
        print("Try using synthetic board for testing by commenting out the port and switching board type")
        return

    session = bml.exg_stream(board_fd, length=window_length, duration=None)

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
        while session.is_running():
            time.sleep(0.01)
            buf: Deque[List[float | str]] = session.get_buffer()
            # if not buf:
            #     continue
            x = list(range(len(buf)))
            
            channel_traces: List[List[float]] = [[float(sample[i+1]) for sample in buf] for i in range(num_ch)]
            for line, vals in zip(lines, channel_traces):
                line.set_data(x, vals)
            all_vals = [v for vals in channel_traces for v in vals]
            if all_vals:
                ax.set_ylim(min(all_vals) * 1.1, max(all_vals) * 1.1)
            
            fig.canvas.draw()
            fig.canvas.flush_events()

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