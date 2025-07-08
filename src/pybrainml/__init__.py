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
from typing import List, Any, Callable, Deque, Optional, Tuple, Dict, Union
from contextlib import contextmanager
# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
# import pandas as pd
# import numpy as np
import numpy as np
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

    def setup(self, name: str, age: int, sex: str):
        """Setup subject information with automatic name hashing for privacy."""
        self.name = self._hash_name(name)
        self.age = age
        self.sex = sex
    
    def _hash_name(self, name: str) -> str:
        """Hash the name for privacy protection."""
        return hashlib.sha256(name.lower().encode()).hexdigest()

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "age": self.age,
            "sex": self.sex
        }

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
    Streams EXG data from the board. Maintains a sliding window of the most recent EXG data (deque of size `length`).
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

def load_experiment_from_json(filepath: str):
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    exp = create_experiment()
    # Load metadata
    meta = data.get("metadata", {})
    # Remove reserved keys if present
    meta_clean = {k: v for k, v in meta.items() if not k.startswith("_")}
    # Subject
    subj = meta_clean.get("subject_info", {})
    exp.metadata.subject_info = Subject(**subj)
    # Description
    exp.metadata.description = meta_clean.get("description")
    # Hardware
    hw = meta_clean.get("hardware_info", {})
    exp.metadata.hardware_info = Hardware(**hw)
    # Placements, Z, Z_REF
    exp.metadata.placements = meta_clean.get("placements", [])
    exp.metadata.Z = meta_clean.get("Z", [])
    exp.metadata.Z_REF = meta_clean.get("Z_REF", 0.0)
    # Load frames
    exp.frames = [Frame(**frame) for frame in data.get("frames", [])]
    return exp

def compute_fft(data: List[List[float]]) -> List[List[float]]:
    """
    Computes the Fast Fourier Transform (FFT) for each row in the data.
    Each row is expected to be a list of float values.
    Returns a list of lists containing the FFT results.
    """
    import numpy as np
    return [np.fft.fft(row).tolist() for row in data]

def compute_power_spectrum(data: List[List[float]]) -> List[List[float]]:
    """
    Computes the Power Spectrum for each row in the data.
    Each row is expected to be a list of float values.
    Returns a list of lists containing the Power Spectrum results.
    """
    import numpy as np
    return [(np.abs(np.fft.fft(row))**2).tolist() for row in data]

def compute_band_powers(data: List[List[float]], bands: List[Tuple[float, float]]) -> List[List[float]]:
    """
    Computes the band powers for each row in the data.
    Each row is expected to be a list of float values.
    Bands should be a list of tuples, where each tuple contains the lower and upper frequency bounds.
    Returns a list of lists containing the band power results.
    """
    import numpy as np
    band_powers = []
    for row in data:
        fft_result = np.fft.fft(row)
        freqs = np.fft.fftfreq(len(row))
        powers = np.abs(fft_result)**2
        band_power_row = []
        for low, high in bands:
            band_mask = (freqs >= low) & (freqs <= high)
            band_power_row.append(np.sum(powers[band_mask]))
        band_powers.append(band_power_row)
    return band_powers


# ===== ELASTICSEARCH INTEGRATION =====

class ElasticsearchClient:
    """
    Elasticsearch client for EEG data storage following TSDS architecture.
    Implements the specification for experiments, frames, and time-series data streams.
    """
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 9200, 
                 username: Optional[str] = None, 
                 password: Optional[str] = None,
                 use_https: bool = False):
        """Initialize Elasticsearch client with connection parameters."""
        self.host = host
        self.port = port
        self.use_https = use_https
        
        scheme = "https" if use_https else "http"
        self.es = Elasticsearch(
            [f"{scheme}://{host}:{port}"],
            basic_auth=(username, password) if username and password else None,
            verify_certs=use_https
        )
        
        if not self.es.ping():
            raise ValueError("Elasticsearch connection failed.")
        
        print(f"Connected to Elasticsearch at {host}:{port}")
    
    def setup_indices(self):
        """Create the experiments and frames indices with proper mappings."""
        
        # Experiments index mapping
        experiments_mapping = {
            "mappings": {
                "properties": {
                    "experiment_id": {"type": "keyword"},
                    "version": {"type": "keyword"},
                    "created_at": {"type": "date"},
                    "created_by": {"type": "keyword"},
                    "subject_info": {
                        "properties": {
                            "subject_id": {"type": "keyword"},
                            "age": {"type": "integer"},
                            "sex": {"type": "keyword"}
                        }
                    },
                    "description": {"type": "text"},
                    "hardware_info": {
                        "properties": {
                            "electrode_type": {"type": "keyword"},
                            "board": {"type": "keyword"},
                            "channels": {"type": "integer"},
                            "sampling_rate": {"type": "integer"}
                        }
                    },
                    "placements": {"type": "keyword"},
                    "Z": {"type": "integer"},
                    "Z_REF": {"type": "integer"}
                }
            }
        }
        
        # Frames index mapping
        frames_mapping = {
            "mappings": {
                "properties": {
                    "experiment_id": {"type": "keyword"},
                    "frame_id": {"type": "keyword"},
                    "label": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "eeg": {
                        "type": "object",
                        "dynamic": True
                    }
                }
            }
        }
        
        # Create indices if they don't exist
        if not self.es.indices.exists(index="experiments"):
            self.es.indices.create(index="experiments", body=experiments_mapping)
            print("Created 'experiments' index")
        
        if not self.es.indices.exists(index="frames"):
            self.es.indices.create(index="frames", body=frames_mapping)
            print("Created 'frames' index")
    
    def setup_tsds(self, channel_names: List[str]):
        """Setup Time Series Data Stream for real-time frame ingestion."""
        
        # Build dynamic EEG field mappings
        eeg_properties = {}
        for channel in channel_names:
            eeg_properties[f"eeg.{channel}"] = {"type": "float"}
        
        template_body = {
            "index_patterns": ["frames-tsds*"],
            "data_stream": {
                "timestamp_field": "@timestamp",
                "index_mode": "time_series"
            },
            "template": {
                "mappings": {
                    "properties": {
                        "@timestamp": {"type": "date"},
                        "experiment_id": {"type": "keyword"},
                        "frame_id": {"type": "keyword"},
                        "label": {"type": "keyword"},
                        **eeg_properties
                    }
                },
                "settings": {
                    "index.lifecycle.name": "frames-tsds-ilm",
                    "index.routing_path": ["experiment_id"],
                    "index.sort.field": ["experiment_id", "@timestamp"],
                    "index.sort.order": ["asc", "asc"]
                }
            }
        }
        
        # Create index template
        self.es.indices.put_index_template(
            name="frames-tsds-template",
            body=template_body
        )
        
        # Create data stream
        try:
            self.es.indices.create_data_stream(name="frames-tsds")
            print("Created 'frames-tsds' data stream")
        except Exception as e:
            if "resource_already_exists_exception" not in str(e):
                raise e
    
    def upload_experiment_metadata(self, experiment_dict: Dict) -> str:
        """Upload experiment metadata and return experiment_id."""
        experiment_id = str(uuid.uuid4())
        
        metadata = experiment_dict.get("metadata", {})
        
        # Prepare experiment document
        experiment_doc = {
            "experiment_id": experiment_id,
            "version": metadata.get("_version", VERSION),
            "created_at": metadata.get("_created_at", datetime.now().isoformat()),
            "created_by": metadata.get("_created_by", "pybrainml"),
            "subject_info": {
                "subject_id": metadata.get("subject_info", {}).get("name", "unknown"),
                "age": metadata.get("subject_info", {}).get("age"),
                "sex": metadata.get("subject_info", {}).get("sex")
            },
            "description": metadata.get("description"),
            "hardware_info": metadata.get("hardware_info", {}),
            "placements": metadata.get("placements", []),
            "Z": metadata.get("Z", []),
            "Z_REF": metadata.get("Z_REF", 0.0)
        }
        
        # Upload to experiments index
        result = self.es.index(
            index="experiments",
            id=experiment_id,
            document=experiment_doc
        )
        
        print(f"Uploaded experiment metadata with ID: {experiment_id}")
        return experiment_id
    
    def upload_frames_bulk(self, experiment_id: str, frames: List[Dict], 
                          channel_names: List[str]) -> List[str]:
        """Upload frames in bulk to the frames index."""
        actions = []
        frame_ids = []
        
        for frame in frames:
            frame_id = str(uuid.uuid4())
            frame_ids.append(frame_id)
            
            # Convert eeg_data to channel-based format
            eeg_data = frame.get("eeg_data", [])
            eeg_dict = {}
            
            if eeg_data and len(channel_names) > 0:
                # Assume eeg_data is a list of lists, take the first sample per channel
                for i, channel in enumerate(channel_names):
                    if i < len(eeg_data) and len(eeg_data[i]) > 0:
                        eeg_dict[channel] = float(eeg_data[i][0])
            
            frame_doc = {
                "experiment_id": experiment_id,
                "frame_id": frame_id,
                "label": frame.get("label"),
                "timestamp": frame.get("timestamp", datetime.now().isoformat()),
                "eeg": eeg_dict
            }
            
            actions.append({
                "_index": "frames",
                "_id": frame_id,
                "_source": frame_doc
            })
        
        if actions:
            helpers.bulk(self.es, actions)
            print(f"Uploaded {len(actions)} frames to 'frames' index")
        
        return frame_ids
    
    def upload_frame_to_tsds(self, experiment_id: str, frame: Dict, 
                           channel_names: List[str]) -> str:
        """Upload a single frame to the TSDS for real-time streaming."""
        frame_id = str(uuid.uuid4())
        
        # Convert eeg_data to flattened channel format for TSDS
        eeg_data = frame.get("eeg_data", [])
        tsds_doc = {
            "@timestamp": frame.get("timestamp", datetime.now().isoformat()),
            "experiment_id": experiment_id,
            "frame_id": frame_id,
            "label": frame.get("label")
        }
        
        # Add flattened EEG data
        if eeg_data and len(channel_names) > 0:
            for i, channel in enumerate(channel_names):
                if i < len(eeg_data) and len(eeg_data[i]) > 0:
                    tsds_doc[f"eeg.{channel}"] = float(eeg_data[i][0])
        
        # Upload to TSDS
        result = self.es.index(
            index="frames-tsds",
            document=tsds_doc
        )
        
        return frame_id
    
    def get_experiment_metadata(self, experiment_id: str) -> Dict:
        """Retrieve experiment metadata by ID."""
        try:
            result = self.es.get(index="experiments", id=experiment_id)
            return result["_source"]
        except Exception as e:
            print(f"Failed to retrieve experiment {experiment_id}: {e}")
            raise
    
    def get_experiment_frames(self, experiment_id: str, 
                            start_time: Optional[str] = None,
                            end_time: Optional[str] = None) -> List[Dict]:
        """Retrieve frames for an experiment with optional time filtering."""
        query = {
            "bool": {
                "must": [
                    {"term": {"experiment_id": experiment_id}}
                ]
            }
        }
        
        if start_time or end_time:
            time_range = {}
            if start_time:
                time_range["gte"] = start_time
            if end_time:
                time_range["lte"] = end_time
            
            query["bool"]["must"].append({
                "range": {"timestamp": time_range}
            })
        
        search_body = {
            "query": query,
            "sort": [{"timestamp": "asc"}],
            "size": 10000  # Adjust as needed
        }
        
        result = self.es.search(index="frames", body=search_body)
        return [hit["_source"] for hit in result["hits"]["hits"]]
    
    def search_tsds_frames(self, experiment_id: str, 
                          start_time: Optional[str] = None,
                          end_time: Optional[str] = None) -> List[Dict]:
        """Search TSDS frames with time range filtering."""
        query = {
            "bool": {
                "must": [
                    {"term": {"experiment_id": experiment_id}}
                ]
            }
        }
        
        if start_time or end_time:
            time_range = {}
            if start_time:
                time_range["gte"] = start_time
            if end_time:
                time_range["lte"] = end_time
            
            query["bool"]["must"].append({
                "range": {"@timestamp": time_range}
            })
        
        search_body = {
            "query": query,
            "sort": [{"@timestamp": "asc"}],
            "size": 10000
        }
        
        result = self.es.search(index="frames-tsds", body=search_body)
        return [hit["_source"] for hit in result["hits"]["hits"]]


def create_elasticsearch_client(host: str = "localhost", 
                              port: int = 9200,
                              username: Optional[str] = None,
                              password: Optional[str] = None,
                              use_https: bool = False) -> ElasticsearchClient:
    """Create and return an Elasticsearch client instance."""
    return ElasticsearchClient(host, port, username, password, use_https)


def upload_experiment_to_elasticsearch(experiment_dict: Dict,
                                     es_client: ElasticsearchClient,
                                     channel_names: Optional[List[str]] = None) -> Dict[str, Union[str, int, List[str]]]:
    """
    Upload complete experiment to Elasticsearch using the TSDS architecture.
    
    Returns:
        Dictionary with experiment_id and upload statistics
    """
    # Setup indices if not already done
    es_client.setup_indices()
    
    # Determine channel names from hardware info or use defaults
    if not channel_names:
        hardware_info = experiment_dict.get("metadata", {}).get("hardware_info", {})
        num_channels = hardware_info.get("channels", 4)
        channel_names = [f"Ch{i+1}" for i in range(num_channels)]
    
    # Setup TSDS with channel names
    es_client.setup_tsds(channel_names)
    
    # Upload metadata
    experiment_id = es_client.upload_experiment_metadata(experiment_dict)
    
    # Upload frames
    frames = experiment_dict.get("frames", [])
    frame_ids = []
    
    if frames:
        frame_ids = es_client.upload_frames_bulk(experiment_id, frames, channel_names)
    
    return {
        "experiment_id": experiment_id,
        "frames_uploaded": len(frame_ids),
        "frame_ids": frame_ids[:10]  # Return first 10 for reference
    }

class RealTimeEEGStreamer:
    """
    Real-time EEG streaming to Elasticsearch TSDS.
    Handles live data ingestion with minimal latency.
    """
    
    def __init__(self, es_client: ElasticsearchClient, experiment_id: str, 
                 channel_names: List[str], buffer_size: int = 100):
        self.es_client = es_client
        self.experiment_id = experiment_id
        self.channel_names = channel_names
        self.buffer_size = buffer_size
        self.frame_buffer = []
        self.is_streaming = False
        
    def start_streaming(self):
        """Start the real-time streaming."""
        self.is_streaming = True
        print(f"Started real-time streaming for experiment {self.experiment_id}")
    
    def stop_streaming(self):
        """Stop streaming and flush remaining buffer."""
        self.is_streaming = False
        if self.frame_buffer:
            self._flush_buffer()
        print("Stopped real-time streaming")
    
    def add_frame(self, frame_data: Dict):
        """Add a frame to the streaming buffer."""
        if not self.is_streaming:
            return
            
        self.frame_buffer.append(frame_data)
        
        if len(self.frame_buffer) >= self.buffer_size:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush the current buffer to Elasticsearch TSDS."""
        if not self.frame_buffer:
            return
            
        try:
            for frame in self.frame_buffer:
                self.es_client.upload_frame_to_tsds(
                    self.experiment_id, frame, self.channel_names
                )
            
            print(f"Flushed {len(self.frame_buffer)} frames to TSDS")
            self.frame_buffer.clear()
            
        except Exception as e:
            print(f"Failed to flush buffer to TSDS: {e}")

def create_realtime_streamer(es_client: ElasticsearchClient, 
                           experiment_dict: Dict,
                           channel_names: Optional[List[str]] = None) -> RealTimeEEGStreamer:
    """
    Create a real-time EEG streamer for live data ingestion.
    
    Args:
        es_client: Elasticsearch client instance
        experiment_dict: Experiment metadata
        channel_names: List of EEG channel names
        
    Returns:
        RealTimeEEGStreamer instance
    """
    # Upload experiment metadata first
    experiment_id = es_client.upload_experiment_metadata(experiment_dict)
    
    # Determine channel names
    if not channel_names:
        hardware_info = experiment_dict.get("metadata", {}).get("hardware_info", {})
        num_channels = hardware_info.get("channels", 4)
        channel_names = [f"Ch{i+1}" for i in range(num_channels)]
    
    # Setup TSDS
    es_client.setup_tsds(channel_names)
    
    return RealTimeEEGStreamer(es_client, experiment_id, channel_names)
