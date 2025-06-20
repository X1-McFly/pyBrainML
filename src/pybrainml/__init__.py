import platform
import time
from datetime import datetime
import hashlib
from enum import Enum
import os
import json
from collections import deque
import threading
import queue
import multiprocessing as mp
from dataclasses import dataclass, field, asdict
from typing import List, Any, Callable, Deque, List, Optional, Tuple
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import pandas as pd
import numpy as np

VERSION = "0.2.9"

class Boards(Enum):

    OpenBCI_Ganglion = BoardIds.GANGLION_BOARD.value
    OpenBCI_Cyton = BoardIds.CYTON_BOARD.value
    OpenBCI_Cyton_Daisy = BoardIds.CYTON_DAISY_BOARD.value
    # BIOCOM_BrainWave1 = BoardIds.CALLIBRI_EEG_BOARD.value

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

    # eeg_channels: list[int] | None = None
    # accel_channels: list[int] | None = None
    # num_rows: int | None = None

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

@contextmanager
def ganglion_session(port: str):
    params = BrainFlowInputParams()
    params.serial_port = port
    board_id = BoardIds.GANGLION_BOARD.value
    board = BoardShim(board_id, params)
    try:
        board.prepare_session()
        board.start_stream()
        yield board
    finally:
        board.stop_stream()
        board.release_session()

def init_ganglion(port: str) -> Tuple[int, int, List[int], int]:
    params = BrainFlowInputParams(); params.serial_port = port
    board_id = BoardIds.GANGLION_BOARD.value
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    ts_channel = BoardShim.get_timestamp_channel(board_id)
    return board_id, sampling_rate, eeg_channels, ts_channel

def start_eeg_stream(
    port: str,
    save_to: Optional[str] = None,
    callback: Optional[Callable[[Deque[List[float]]], None]] = None,
    length: Optional[int] = 200,
) -> None:

    board_id, sampling_rate, eeg_chs, ts_ch = init_ganglion(port)
    buf: Deque[List[float]] = deque(maxlen=length)
    q: queue.Queue = queue.Queue()
    executor = ThreadPoolExecutor(max_workers=1)

    if save_to:
        def writer():
            with open(save_to, "a") as f:
                batch = []
                while True:
                    data = q.get()
                    if data is None:
                        break
                    batch.append(data)
                    if len(batch) >= 10:
                        f.writelines(json.dumps(r) + "\n" for r in batch)
                        batch.clear()
        threading.Thread(target=writer, daemon=True).start()

    print("Streaming EEG... Ctrl+C to stop\n")
    try:
        with ganglion_session(port) as board:
            while True:
                data = board.get_board_data()
                if data.shape[1] == 0:
                    time.sleep(0.001)
                    continue

                timestamps = pd.to_datetime(data[ts_ch], unit="s")
                eeg = data[eeg_chs]
                for i in range(min(len(timestamps), eeg.shape[1])):
                    sample = [float(eeg[ch][i]) for ch in range(len(eeg_chs))]
                    print(f"[{', '.join(f'{v:.2f}' for v in sample)}]")
                    buf.append(sample)
                    if save_to:
                        q.put(sample)
                    if callback:
                        executor.submit(callback, buf)
    except KeyboardInterrupt:
        print("Streaming interrupted.")
    finally:
        if save_to:
            q.put(None)
        executor.shutdown(wait=False)

def get_unique_file(fd) -> str:
    base, ext = os.path.splitext(fd)
    final_fd = base + ext
    counter = 1
    while os.path.exists(final_fd):
        final_fd = f"{base} ({counter}){ext}"
        counter += 1
    return final_fd

def load_ndjson(fd) -> list[Any]:
    with open(fd, "r") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    return entries

def post_process() -> None:
    recorded_data = load_ndjson("test.ndjson")
    # print(recorded_data)



    return