#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG Streaming, Real-Time Plotting, and Gamma-Band Power Extraction Example

Author: Martin McCorkle
Date: 2025-06-30
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

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from yaspin import yaspin

import pybrainml as bml
from pybrainml import ElectrodeType, Boards, Frame


def main():
    port = "COM8"
    window_length = 200  # sliding‐window size

    # Build experiment metadata
    exp = bml.create_experiment()
    exp.metadata.subject_info.setup("John Doe", 35, "F")
    exp.metadata.hardware_info.setup(ElectrodeType.HYBRID, Boards.OpenBCI_Ganglion)

    # Start streaming in background; data saved to out.ndjson
    session = bml.exg_stream(port=port, save_to="out.ndjson", length=window_length)
    print(f"Streaming started on {port}")

    # Determine sampling rate and EEG channels
    _, sampling_rate, eeg_chs = bml.init_board(port)
    num_ch = len(eeg_chs)

    # Prepare real‐time plots: raw EEG and gamma‐band power
    plt.ion()
    fig, (ax_raw, ax_gamma) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    # Raw EEG plot
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

    try:
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

    except KeyboardInterrupt:
        print("Stopping session…")

    finally:
        # Persist experiment metadata
        with open("test.json", "w") as f:
            json.dump(exp.to_dict(), f, indent=4)
        plt.ioff()
        plt.close()
        print("Session stopped; metadata saved to test.json")


if __name__ == "__main__":
    main()
