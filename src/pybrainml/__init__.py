import platform
import time
from datetime import datetime
import hashlib
from enum import Enum
import os
import json
from collections import deque
from threading import Thread
import uuid
# import queue
from multiprocessing import Process, Queue
from dataclasses import dataclass, field, asdict
from typing import List, Any, Callable, Deque, List, Optional, Tuple
from contextlib import contextmanager
# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
# import pandas as pd
# import numpy as np
from yaspin import yaspin
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers

VERSION = "0.3.2"

BoardShim.disable_board_logger() 

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
    eeg_data: List[List[Any]] = field(default_factory=list)

    @classmethod
    def create(cls, lbl: Optional[str], eg: List[Any]):
        return cls(label=lbl, eeg_data=eg)

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
        self.sampling_rate = BoardShim.get_sampling_rate(board_name.board_id)
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

        def user_setup(self, name: str, age: Optional[int], sex: Optional[str]):
            self.metadata.subject_info = Subject(name, age, sex)

        def hardware_setup(self, electrode_type: ElectrodeType, board_name: Boards):
            hw = Hardware()
            hw.setup(electrode_type, board_name)
            self.metadata.hardware_info = hw

        def to_dict(self):
            return asdict(self)
    
    return Experiment()

def connect_board(port: str, 
                  board_id: Boards, 
                  max_retries: int = 3) -> BoardShim:

    params = BrainFlowInputParams()
    params.serial_port = port   
    board_fd = BoardShim(board_id.value, params)

    def attempt_connect():
        try:
            board_fd.prepare_session()
            board_fd.release_session()
            return True
        except Exception:
            return False

    os.system('cls' if os.name == 'nt' else 'clear')
    with yaspin(text="Connecting to board...", color="green") as spinner:
        for attempt in range(1, max_retries + 1):
            if attempt_connect():
                spinner.text = ""
                spinner.color = "green"
                spinner.ok("Connected")
                break
            elif attempt < max_retries:
                spinner.text = f"Attempt {attempt}/{max_retries} failed..."
                spinner.color = "yellow"
            else:
                spinner.text = ""
                spinner.color = "red"
                spinner.fail("Connection failed")

    @staticmethod
    def check_connection():
        pass

    # ts_channel = BoardShim.get_timestamp_channel(board_id)
    return board_fd

@contextmanager
def board_session(board_fd: BoardShim):
    try:
        board_fd.prepare_session()
        board_fd.start_stream()
        yield board_fd
    finally:
        board_fd.stop_stream()
        board_fd.release_session()

def _save_worker(path: str, q: Queue):
    if os.path.exists(path):
        os.remove(path)
    with open(path, "a", buffering=1) as f:
        while True:
            batch = q.get()
            if batch is None:
                break
            for item in batch:
                f.write(json.dumps(item) + "\n")
            f.flush()

def exg_stream(
    board_fd: BoardShim,
    length: int = 200,
    save_fd: str = "data",
    duration: Optional[float] = None,
):
    """
    Streams EEG data from the board. Maintains a sliding window of the most recent EEG data (deque of size `length`).
    The program uses two alternating buffers to batch-save data to disk efficiently in a background thread.
    Returns a handle with a `get_buffer()` and `stop()` method.
    """
    from threading import Event

    os.makedirs(save_fd, exist_ok=True)
    temp_f = os.path.join(os.getcwd(), save_fd, "temp.ndjson")
    
    # board_fd, sampling_rate, eeg_chs = connect_board(port, Boards.OpenBCI_Ganglion.value)
    buf: Deque[List[float | str]] = deque(maxlen=length)
    buffers = [[], []]
    active_idx = 0
    max_buf_size = length
    stop_event = Event()

    save_q = Queue()
    save_thread = Thread(target=_save_worker, args=(temp_f, save_q), daemon=True)
    save_thread.start()

    def save_sample(sample):
        nonlocal active_idx
        current_buf = buffers[active_idx]
        current_buf.append(sample)
        if len(current_buf) >= max_buf_size:
            if save_q:
                save_q.put(current_buf.copy())
            buffers[active_idx] = []
            active_idx = 1 - active_idx

    def _run_stream():
        eeg_chs = BoardShim.get_eeg_channels(board_fd.board_id)
        try:
            with board_session(board_fd) as board:
                start_time = time.time()
                while not stop_event.is_set() and (duration is None or time.time() - start_time < duration):
                    data = board.get_board_data()
                    if data.shape[1] == 0:
                        time.sleep(0.001)
                        continue
                    eeg_data = data[eeg_chs]
                    for i in range(eeg_data.shape[1]):
                        sample = [datetime.now().isoformat()] + [float(eeg_data[ch][i]) for ch in range(len(eeg_chs))]
                        buf.append(sample)
                        save_sample(sample)
        finally:
            if save_q:
                for b in buffers:
                    if b:
                        save_q.put(b)
                save_q.put(None)
                if save_thread:
                    save_thread.join()

    stream_thread = Thread(target=_run_stream, daemon=True)

    class Handle:
        @staticmethod
        def start() -> None:
            if not stream_thread.is_alive():
                stream_thread.start()
        
        @staticmethod
        def get_buffer() -> Deque[List[float | str]]:
            snapshot = buf.copy()
            return deque(item.copy() for item in snapshot)
        
        @staticmethod
        def stop() -> None:
            stop_event.set()
            if save_q:
                save_q.put(None)

        @staticmethod
        def is_running() -> bool:
            return stream_thread.is_alive()
        
        # remove later
        @staticmethod
        def eeg_channels() -> List[int]:
            return BoardShim.get_eeg_channels(board_fd.board_id)
        
        @staticmethod
        def get_final():
            return post_process(temp_f)

    return Handle()

def get_unique_file(dir, fd) -> str:
    base, ext = os.path.splitext(fd)
    final_fd = base + ext
    counter = 1
    while os.path.exists(os.path.join(dir, final_fd)):
        final_fd = f"{base}({counter}){ext}"
        counter += 1
    return final_fd

def post_process(fd) -> Frame | None:
    with open(fd, "r") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if not entries:
        print(f"No valid entries found in {fd}")
        return
    if os.path.exists(fd):
            os.remove(fd)
    frame = Frame(
        label=None,
        timestamp=entries[0][0] if entries else None,
        eeg_data=[[entry[0]] + entry[1:] for entry in entries],
    )
    return frame

def export_experiment(session, 
                      exp, 
                      data_dir: str = "data"):
    
    filename = os.path.join(data_dir, get_unique_file(data_dir, "test.json"))
    processed_frame = session.get_final()
    if processed_frame is not None:
        print(f"Saving processed frame to {filename}...")
        exp.frames.append(processed_frame)
        with open(filename, "w") as f:
            json.dump(exp.to_dict(), f, indent=4)
    else:
        print("No processed frame to save.")

@dataclass
class placements:
    pass

@dataclass
class upload:

    index: str = "biocom"
    host: str = "localhost"
    port: int = 9200
    username: Optional[str] = None
    password: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None

    def __post_init__(self):
        if not self.index:
            raise ValueError("Index name must be provided")
        if not self.host:
            raise ValueError("Host must be provided")
        if not isinstance(self.port, int) or self.port <= 0:
            raise ValueError("Port must be a positive integer")

    @staticmethod
    def es_connect():
        dotenv_path = os.path.join(os.getcwd(),"env", 'keys.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)
            print("Loading environment variables...")
        else:
            raise FileNotFoundError(f"Environment file not found at {dotenv_path}")

        ES_HOST = os.getenv("ES_HOST")
        ES_ID = os.getenv("ES_ID")
        ES_SECRET = os.getenv("ES_SECRET")

        if not all([ES_HOST, ES_ID, ES_SECRET]):
            raise ValueError("Missing required environment variables: ES_HOST, ES_ID, ES_SECRET")

        try:
            es = Elasticsearch([ES_HOST], api_key=(ES_ID, ES_SECRET), verify_certs=True) # type: ignore
            if not es.ping():
                raise ValueError("Ping failed: could not connect to cluster.")
            print("Connected to Elasticsearch successfully!")
        except Exception as e:
            print(f"[!] Elasticsearch connection failed:\n{e}")
            raise


# def elastic_search_upload(
#     fd: str,
#     index: str,
#     host: str = "localhost",
#     port: int = 9200,
#     username: Optional[str] = None,
#     password: Optional[str] = None,
# ):
#     from elasticsearch import Elasticsearch, helpers

#     es = Elasticsearch(
#         [{"host": host, "port": port}],
#         http_auth=(username, password) if username and password else None,
#     )

#     with open(fd, "r") as f:
#         entries = [json.loads(line) for line in f if line.strip()]

#     if not entries:
#         print(f"No valid entries found in {fd}")
#         return

#     actions = [
#         {
#             "_index": index,
#             "_source": entry,
#         }
#         for entry in entries
#     ]

#     helpers.bulk(es, actions)
#     print(f"Uploaded {len(actions)} entries to index '{index}'")
