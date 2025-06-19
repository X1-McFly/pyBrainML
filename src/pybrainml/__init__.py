import platform
import time
from datetime import datetime
import hashlib
from enum import Enum
import os
import json
from collections import deque
from threading import Thread
import threading
import queue
import multiprocessing as mp
from dataclasses import dataclass, field, asdict
from typing import List, Any

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import pandas as pd
import numpy as np

VERSION = "0.2.8"

# change to native BrainFlow
class Boards(Enum):
    OpenBCI_Ganglion = ("OpenBCI Ganglion", 4, 200)
    OpenBCI_Cyton = ("OpenBCI Cyton", 8, 250)
    OpenBCI_Cyton_Daisy = ("OpenBCI Cyton Daisy", 16, 125)
    BIOCOM_BrainWave1 = ("BIOCOM BrainWave1", 4, 250)

    def __init__(self, display_name: str, channels: int, sampling_rate: int):
        self.display_name = display_name
        self._channels = channels
        self._sampling_rate = sampling_rate

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate
    
class ElectrodeType(Enum):
            HYBRID = "Hybrid"
            WET = "Wet"
            DRY = "Dry"

@dataclass
class Subject:
    name: str | None = None
    age: int | None = None
    sex: str | None = None

    def setup(self, NAME, AGE, SEX):
        self.name = NAME
        self.age = AGE
        self.sex = SEX
    
    def hash_name(self):
        if self.name is None:
            raise ValueError("Name must be set before hashing")
        else:
            self.name = hashlib.sha256(self.name.lower().encode()).hexdigest()

    def __setattr__(self, key, value):
        if key == "name" and value is not None:
            value = hashlib.sha256(value.lower().encode()).hexdigest()
        super().__setattr__(key, value)

    def to_dict(self):
        return asdict(self)

@dataclass
class Frame:
    label: str | None = None
    timestamp: str | None = None
    eeg_data: List[Any] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

@dataclass  
class Hardware:
    electrode_type: str | None = None

    board: str | None = None
    channels: int | None = None
    
    sampling_rate: int | None = None

    def setup(self, electype: ElectrodeType, board_name: Boards):
        self.electrode_type = electype.name
        self.board = board_name.name
        self.channels = board_name.channels
        self.sampling_rate = board_name.sampling_rate

    def to_dict(self):
        return asdict(self)

def create_experiment():

    @dataclass
    class Metadata:
        _version: str = VERSION
        _created_at: str = datetime.now().isoformat()
        _created_by: str = platform.node()
        subject_info: Subject = field(default_factory=Subject)
        description: str | None = None

        hardware_info: Hardware = field(default_factory=Hardware)

        placements: List[str] = field(default_factory=list)
        Z: List[float] = field(default_factory=list)
        Z_REF: float = field(default_factory=float)

        def to_dict(self):
            return asdict(self)

    @dataclass
    class Experiment:
        metadata: Metadata = field(default_factory=Metadata)
        frames: List[Frame] = field(default_factory=list)

        def to_dict(self):
            return asdict(self)

    return Experiment()

def init_ganglion(port):
    params = BrainFlowInputParams()
    params.serial_port = port

    board_id = BoardIds.GANGLION_BOARD.value
    board = BoardShim(board_id, params)

    sampling_rate = BoardShim.get_sampling_rate(board_id)
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    timestamp_channel = BoardShim.get_timestamp_channel(board_id)
    # channel_names = [f"Ch{i+1}" for i in range(len(eeg_channels))]

    return board, sampling_rate, eeg_channels, timestamp_channel

def start_eeg_stream(board, eeg_channels, timestamp_channel, save=False, foo=None):
    
    print("Streaming EEG data... Press Ctrl+C to stop.\n")
    board.prepare_session()
    board.start_stream()
    
    buffer = deque(maxlen=200)
    q = queue.Queue()
    writer_thread = None

    buffer1 = np.array(100, dtype=np.float32)
    buffer2 = np.array(100, dtype=np.float32)

    if save:
        meta_fd = get_unique_file("data/test_blocks/blocks.json")
        eeg_fd = get_unique_file("data/test_blocks/eeg.ndjson")
        print(meta_fd)
        print(eeg_fd)
        writer_thread = threading.Thread(target=start_file_writer, args=(eeg_fd, q), daemon=True)
        writer_thread.start()

    try:
        while True:
            data = board.get_board_data()

            if data.shape[1] == 0:
                time.sleep(0.001)
                continue

            timestamps = pd.to_datetime(data[timestamp_channel], unit='s')
            eeg_data = data[eeg_channels]

            for i in range(min(len(timestamps), eeg_data.shape[1])):
                row = [float(eeg_data[ch][i]) for ch in range(len(eeg_channels))]
                print(f"[{','.join(f'{val:.2f}' for val in row)}]")

                buffer.append(row)
                if save:
                    q.put(row)
                if foo:
                    t = threading.Thread(target=foo, args=(buffer.copy(),))
                    t.daemon = True
                    t.start()

    except KeyboardInterrupt:
        print("Streaming interrupted.")
    finally:
        if save and writer_thread is not None:
            q.put(None)
            writer_thread.join()
        board.stop_stream()
        board.release_session()

def get_unique_file(fd):
    base, ext = os.path.splitext(fd)
    final_fd = base + ext
    counter = 1
    while os.path.exists(final_fd):
        final_fd = f"{base} ({counter}){ext}"
        counter += 1
    return final_fd
