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

# BrainFlow board compatibility wrapper
class Boards(Enum):
    """Enumeration of supported boards with BrainFlow metadata."""

    OpenBCI_Ganglion = BoardIds.GANGLION_BOARD.value
    OpenBCI_Cyton = BoardIds.CYTON_BOARD.value
    OpenBCI_Cyton_Daisy = BoardIds.CYTON_DAISY_BOARD.value
    BIOCOM_BrainWave1 = BoardIds.CALLIBRI_EEG_BOARD.value

    def __init__(self, board_id: int):
        self.board_id = board_id
        desc = BoardShim.get_board_descr(board_id)
        self.display_name = desc.get("name", self.name)
        self._sampling_rate = desc.get("sampling_rate", 0)
        self._eeg_channels = desc.get("eeg_channels", [])
        self._accel_channels = desc.get("accel_channels", [])
        self._num_rows = desc.get("num_rows", len(self._eeg_channels))

    @property
    def channels(self) -> int:
        """Number of EEG channels."""
        return len(self._eeg_channels)

    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate

    @property
    def eeg_channels(self) -> list[int]:
        return self._eeg_channels

    @property
    def accel_channels(self) -> list[int]:
        return self._accel_channels

    @property
    def num_rows(self) -> int:
        return self._num_rows

    def to_dict(self) -> dict:
        """Return board information suitable for JSON serialization."""
        return {
            "name": self.display_name,
            "board_id": self.board_id,
            "sampling_rate": self._sampling_rate,
            "eeg_channels": self._eeg_channels,
            "accel_channels": self._accel_channels,
            "num_rows": self._num_rows,
        }
    
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

    eeg_channels: list[int] | None = None
    accel_channels: list[int] | None = None
    num_rows: int | None = None

    def setup(self, electype: ElectrodeType, board_name: Boards):
        self.electrode_type = electype.name
        self.board = board_name.name
        self.channels = board_name.channels
        self.sampling_rate = board_name.sampling_rate
        self.eeg_channels = board_name.eeg_channels
        self.accel_channels = board_name.accel_channels
        self.num_rows = board_name.num_rows

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

def start_file_writer(eeg_fd, q):
    buffer = []
    with open(eeg_fd, "a") as f:
        while True:
            item = q.get()
            if item is None:
                break
            buffer.append(item)
            if len(buffer) >= 10:
                f.writelines(json.dumps(x) + "\n" for x in buffer)
                buffer.clear()

def get_unique_file(fd):
    base, ext = os.path.splitext(fd)
    final_fd = base + ext
    counter = 1
    while os.path.exists(final_fd):
        final_fd = f"{base} ({counter}){ext}"
        counter += 1
    return final_fd
