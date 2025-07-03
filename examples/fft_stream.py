#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG Streaming, Real-Time Plotting, and Gamma-Band Power Extraction Example

Author: Martin McCorkle
Date: 2025-07-03
Description:
    Streams EEG data from a BrainFlow-compatible board using pybrainml,
    saves raw data to an NDJSON file, plots all EEG channels in real time,
    and additionally computes and plots the average gamma-band (30–45 Hz)
    power as a time series.

Dependencies:
    - pybrainml>=0.3
    - numpy
    - scipy
    - matplotlib
    - brainflow
    - yaspin
"""

import time
import json
from collections import deque
from datetime import datetime
from typing import Deque, List, Optional
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
# from yaspin import yaspin

import pybrainml as bml
from pybrainml import ElectrodeType, Boards, Frame


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

    # Prepare real‐time plots: raw EEG and gamma‐band power
    plt.ion()
    fig, (ax_raw, ax_gamma) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    # Raw EEG plot
    num_ch = len(session.eeg_channels())
    raw_lines = [ax_raw.plot([], [])[0] for _ in range(num_ch)]
    ax_raw.set_title("EEG Channels")
    ax_raw.set_ylabel("Amplitude")
    ax_raw.set_xlim(0, window_length)
    ax_raw.legend().set_visible(False)

    # Gamma power plot
    gamma_line, = ax_gamma.plot([], [], color="m")
    ax_gamma.set_title("Average Gamma (30–45 Hz) Power")
    ax_gamma.set_xlabel("Sample Index")
    ax_gamma.set_ylabel("Power")
    ax_gamma.set_xlim(0, window_length)
    ax_gamma.legend().set_visible(False)

    # Buffer for gamma‐power time series
    gamma_buf: Deque[float] = deque(maxlen=window_length)

    # Start experiment
    session.start()

    try:
        # sampling_rate = exp.metadata.hardware_info.sampling_rate
        sampling_rate = 200
        if sampling_rate is None:
            raise ValueError("Sampling rate is None; cannot proceed.")
        sampling_rate = float(sampling_rate)

        while True:
            time.sleep(0.1)
            buf: Deque[List[float | str]] = session.get_buffer()
            if not buf:
                continue

            # Prepare x axis
            x = list(range(len(buf)))

            # Extract raw channel time series
            channel_traces: List[List[float]] = [
                [float(sample[i+1]) for sample in buf] for i in range(num_ch)
            ]
            for line, vals in zip(raw_lines, channel_traces):
                line.set_data(x, vals)

            # Compute and append average gamma‐band power
            data_arr = np.array(channel_traces)  # shape (num_ch, window_length)
            gamma_pows = []
            for ch_data in data_arr:
                freqs, psd = welch(ch_data, fs=sampling_rate, nperseg=min(256, ch_data.size))
                mask = (freqs >= 30) & (freqs <= 45)
                gamma_pows.append(np.trapz(psd[mask], freqs[mask]))
            avg_gamma = float(np.mean(gamma_pows))
            gamma_buf.append(avg_gamma)

            # Update gamma plot
            xg = list(range(len(gamma_buf)))
            gamma_line.set_data(xg, list(gamma_buf))

            # Rescale y‐limits
            all_raw = [v for vals in channel_traces for v in vals]
            if all_raw:
                ax_raw.set_ylim(min(all_raw) * 1.1, max(all_raw) * 1.1)
            if gamma_buf:
                ax_gamma.set_ylim(min(gamma_buf) * 0.9, max(gamma_buf) * 1.1)

            fig.canvas.draw()
            fig.canvas.flush_events()

            if not session.is_running():
                break

    except KeyboardInterrupt:
        pass

    finally:
        print("Stopping session...")
        session.stop()
        
        filename = os.path.join(data_dir, bml.get_unique_file(data_dir, "test.json"))
        processed_frame = session.get_final()
        if processed_frame is not None:
            print(f"Saving processed frame to {filename}...")
            exp.frames.append(processed_frame)
            with open(filename, "w") as f:
                json.dump(exp.to_dict(), f, indent=4)
        else:
            print("No processed frame to save.")
        
        plt.ioff()
        plt.close()
    return

if __name__ == "__main__":
    main()
