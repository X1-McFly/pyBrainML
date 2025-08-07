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
from typing import List, Any, Deque, Optional, Tuple, Dict, Union
from contextlib import contextmanager
import logging
# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
# import pandas as pd
# import numpy as np
import numpy as np
from yaspin import yaspin
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers

VERSION = "0.3.4"

BoardShim.disable_board_logger() 
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s.%(msecs)03d] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class Boards(Enum):

    Synthetic = BoardIds.SYNTHETIC_BOARD.value
    OpenBCI_Ganglion = BoardIds.GANGLION_BOARD.value
    OpenBCI_Cyton = BoardIds.CYTON_BOARD.value
    OpenBCI_Cyton_Daisy = BoardIds.CYTON_DAISY_BOARD.value

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
    ACTIVE = "Active"
    PASSIVE = "Passive"

@dataclass
class Subject:
    name: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None

    def setup(self, name: Optional[str], age: int, sex: Optional[str]):
        self.name = self._hash_name(name) if name else None
        self.age = age
        self.sex = sex
    
    def _hash_name(self, name: str) -> str:
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
    eeg_data: List[float] = field(default_factory=list)
    experiment_id: str | None = None
    frame_id: str | None = None

    @classmethod
    def create(cls, lbl: Optional[str], eg: List[float], exp_id: Optional[str] = None):
        return cls(label=lbl, eeg_data=eg, experiment_id=exp_id, frame_id=str(uuid.uuid4()))

    def to_dict(self):
        return asdict(self)
    
    def to_es_doc(self) -> Dict:
        """Convert frame to Elasticsearch document format"""
        return {
            "experiment_id": self.experiment_id,
            "frame_id": self.frame_id,
            "label": self.label,
            "timestamp": self.timestamp,
            "eeg_data": self.eeg_data
        }


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
        version: str = VERSION
        created_at: str = datetime.now().isoformat()
        created_by: str = platform.node()
        # uuid: str = uuid.uuid4().hex
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

        def user_setup(self, name: Optional[str], age: Optional[int], sex: Optional[str]):
            self.metadata.subject_info = Subject(name, age, sex)

        def hardware_setup(self, electrode_type: ElectrodeType, board_name: Boards):
            hw = Hardware()
            hw.setup(electrode_type, board_name)
            self.metadata.hardware_info = hw

        def to_dict(self):
            return asdict(self)
        
        def to_es_doc(self) -> Dict:
            meta = self.metadata.to_dict()
            return {**meta}
    
    return Experiment()

def connect_board(experiment, 
                  port: Optional[str] = None, 
                  max_retries: int = 3) -> BoardShim:

    # Get board info from experiment's hardware setup
    board_name = experiment.metadata.hardware_info.board
    if not board_name:
        raise ValueError("Hardware not configured. Call hardware_setup() first.")
    
    print(f"Attempting to connect to board: {board_name}")
    
    # Get the board enum from the name
    board_enum = getattr(Boards, board_name)
    print(f"Board ID: {board_enum.board_id}, Display name: {board_enum.display_name}")
    
    params = BrainFlowInputParams()
    
    # Handle synthetic board - no port needed
    if board_enum == Boards.Synthetic:
        print("Using synthetic board - no port needed")
        # Synthetic board doesn't need a port, just create the board
        board_fd = BoardShim(board_enum.board_id, params)
        os.system('cls' if os.name == 'nt' else 'clear')
        with yaspin(text="Initializing synthetic board...", color="green") as spinner:
            try:
                board_fd.prepare_session()
                board_fd.release_session()
                spinner.text = ""
                spinner.color = "green"
                spinner.ok("Synthetic board initialized")
            except Exception as e:
                spinner.text = ""
                spinner.color = "red"
                spinner.fail(f"Synthetic board initialization failed: {e}")
                raise
        return board_fd
    
    # Handle real boards - port required
    if not port:
        raise ValueError(f"Port is required for {board_name} board")
    
    print(f"Using port: {port}")
    params.serial_port = port   
    board_fd = BoardShim(board_enum.board_id, params)

    def attempt_connect():
        try:
            board_fd.prepare_session()
            board_fd.release_session()
            return True
        except Exception as e:
            print(f"Connection attempt failed: {e}")
            return False

    os.system('cls' if os.name == 'nt' else 'clear')
    connection_successful = False
    with yaspin(text="Connecting to board...", color="green") as spinner:
        for attempt in range(1, max_retries + 1):
            if attempt_connect():
                spinner.text = ""
                spinner.color = "green"
                spinner.ok("Connected")
                connection_successful = True
                break
            elif attempt < max_retries:
                spinner.text = f"Attempt {attempt}/{max_retries} failed..."
                spinner.color = "yellow"
            else:
                spinner.text = ""
                spinner.color = "red"
                spinner.fail("Connection failed")
    
    if not connection_successful:
        raise ConnectionError(f"Failed to connect to {board_name} board on {port} after {max_retries} attempts")

    return board_fd

@contextmanager
def board_session(board_fd: BoardShim):
    try:
        # Ensure no existing session is active
        try:
            board_fd.release_session()
        except:
            pass  # Ignore if no session was active
        
        board_fd.prepare_session()
        board_fd.start_stream()
        yield board_fd
    finally:
        try:
            board_fd.stop_stream()
        except:
            pass  # Ignore if stream wasn't started
        try:
            board_fd.release_session()
        except:
            pass  # Ignore if session wasn't prepared

def _save_worker(path: str, q: Queue, es_client=None, 
                experiment_id: Optional[str] = None, use_elasticsearch: bool = False,
                es_index_name: str = "bci_tsds", batch_size: int = 100):
    """
    Worker function that saves data batches to file and/or Elasticsearch.
    Handles both local file storage and real-time ES indexing.
    
    Args:
        path: File path for local storage
        q: Queue containing data batches
        es_client: Elasticsearch client (optional)
        experiment_id: Experiment ID for ES documents
        use_elasticsearch: Whether to use Elasticsearch
        es_index_name: Name of ES index to use
        batch_size: Batch size for ES bulk operations
    """
    if not use_elasticsearch and os.path.exists(path):
        os.remove(path)
    
    # File handle for local storage
    file_handle = None
    if not use_elasticsearch:
        file_handle = open(path, "a", buffering=1)
    
    # Elasticsearch batch management
    es_batch = []
    sequence_counter = 0
    
    # Setup ES index if using Elasticsearch
    if use_elasticsearch and es_client:
        try:
            es.create_tsds_index(es_client, es_index_name)
        except Exception as e:
            logger.error(f"Failed to create ES index: {e}")
            use_elasticsearch = False
    
    try:
        while True:
            batch = q.get()
            if batch is None:
                break
            
            # Save to local file if not using ES exclusively
            if file_handle:
                for item in batch:
                    file_handle.write(json.dumps(item) + "\n")
                file_handle.flush()
            
            # Process for Elasticsearch if enabled
            if use_elasticsearch and es_client and experiment_id:
                for item in batch:
                    try:
                        # Extract EEG channels from the data item
                        # item format: [timestamp, ch1, ch2, ch3, ch4, ...]
                        if len(item) >= 5:  # timestamp + at least 4 channels
                            timestamp = item[0] if isinstance(item[0], str) else datetime.now().isoformat()
                            channels = item[1:5]  # Take first 4 EEG channels
                            
                            # Create ES document
                            doc = es.tsds_mapping(
                                doc_id=experiment_id,
                                ch1=float(channels[0]),
                                ch2=float(channels[1]),
                                ch3=float(channels[2]),
                                ch4=float(channels[3]),
                                timestamp=timestamp,
                                sequence_id=sequence_counter
                            )
                            
                            es_batch.append(doc)
                            sequence_counter += 1
                            
                            # Bulk index when batch is full
                            if len(es_batch) >= batch_size:
                                es.bulk_index_data(es_client, es_batch, es_index_name)
                                es_batch.clear()
                                
                    except Exception as e:
                        logger.error(f"Error processing item for ES: {e}")
                        
        # Upload any remaining ES documents
        if es_batch and use_elasticsearch and es_client:
            es.bulk_index_data(es_client, es_batch, es_index_name)
            
    except Exception as e:
        logger.error(f"Error in save worker: {e}")
    finally:
        if file_handle:
            file_handle.close()

def exg_stream(
    board_fd: BoardShim,
    length: int = 200,
    save_fd: str = "data",
    duration: Optional[float] = None,
    use_elasticsearch: bool = False,
    es_index_name: str = "bci_tsds",
    experiment_id: Optional[str] = None,
    es_batch_size: int = 100,
    experiment = None,
    es_meta_index: str = "bci_experiments",
):
    """
    Streams EXG data from the board. Maintains a sliding window of the most recent EXG data (deque of size `length`).
    The program uses two alternating buffers to batch-save data to disk efficiently in a background thread.
    
    Args:
        board_fd: BoardShim instance for the connected board
        length: Size of the sliding window buffer
        save_fd: Directory to save data files
        duration: Optional duration to stream for (seconds)
        use_elasticsearch: Whether to enable Elasticsearch storage
        es_index_name: Name of Elasticsearch index
        experiment_id: Experiment ID for Elasticsearch documents
        es_batch_size: Batch size for Elasticsearch bulk operations
        experiment: Experiment object for metadata indexing
        es_meta_index: Name of Elasticsearch metadata index
        
    Returns:
        Handle with get_buffer(), stop(), start(), and other control methods.
    """
    from threading import Event

    os.makedirs(save_fd, exist_ok=True)
    temp_f = os.path.join(os.getcwd(), save_fd, "temp.ndjson")

    # === ELASTICSEARCH SETUP (Comment out these lines to disable) ===
    es_client = es.create_es_client() if use_elasticsearch else None
    es_doc_id = None
    if es_client:
        if not experiment_id: experiment_id = str(uuid.uuid4())
        es.create_tsds_index(es_client, es_index_name)
        # Index experiment metadata if experiment object provided
        if experiment:
            es_doc_id = es.index_experiment_metadata(es_client, experiment, es_meta_index)
            if es_doc_id: experiment_id = es_doc_id  # Use ES doc ID as experiment ID
    # === END ELASTICSEARCH SETUP ===
    
    buf: Deque[List[float | str]] = deque(maxlen=length)
    buffers = [[], []]
    active_idx = 0
    max_buf_size = length
    stop_event = Event()

    save_q = Queue()
    save_thread = Thread(
        target=_save_worker, 
        args=(temp_f, save_q, es_client, experiment_id, use_elasticsearch, es_index_name, es_batch_size), 
        daemon=True
    )
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

    @staticmethod
    def is_streaming() -> bool:
        """
        Check if data is still streaming from the device.
        Returns True if device is connected and streaming data, False otherwise.
        """
        if not stream_thread.is_alive():
            return False
        
        try:
            current_size = len(buf)
            if current_size == 0:
                return False
            
            time.sleep(0.1)
            new_size = len(buf)
            
            # If buffer size changed, data is streaming
            return new_size != current_size or current_size == buf.maxlen
        except Exception:
            return False

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
        
        @staticmethod
        def eeg_channels() -> List[int]:
            return BoardShim.get_eeg_channels(board_fd.board_id)
        
        @staticmethod
        def get_final():
            return post_process(temp_f)
        
        @staticmethod
        def get_experiment_id() -> Optional[str]:
            return experiment_id
        
        @staticmethod
        def get_es_client():
            return es_client
        
        @staticmethod
        def is_using_elasticsearch() -> bool:
            return use_elasticsearch and es_client is not None
        
        @staticmethod
        def get_es_doc_id() -> Optional[str]:
            return es_doc_id

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
    
    # Convert to new Frame format - eeg_data as simple list
    frame = Frame(
        label=None,
        timestamp=entries[0][0] if entries else None,
        eeg_data=entries[-1][1:] if entries else [],  # Take last sample's EEG data
        frame_id=str(uuid.uuid4())
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

def debug_es_plot(index_name="bci_tsds", size=200, experiment_id=None, realtime=False):
    """
    Convenience function to debug plot Elasticsearch data
    
    Args:
        index_name: ES index name (default: "bci_tsds")
        size: Number of recent documents to plot (default: 200)
        experiment_id: Filter by specific experiment ID (optional)
        realtime: If True, creates real-time updating plot (default: False)
    
    Returns:
        Plot figure and axis objects
    
    Example:
        # Static plot of last 200 samples
        bml.debug_es_plot()
        
        # Real-time plot
        bml.debug_es_plot(realtime=True)
        
        # Plot specific experiment
        bml.debug_es_plot(experiment_id="your-experiment-id")
    """
    try:
        client = es.create_es_client()
        return es.debug_plot_es_data(client, index_name, size, experiment_id, realtime)
    except Exception as e:
        logger.error(f"Failed to create debug plot: {e}")
        print(f"Error: {e}")
        print("Make sure Elasticsearch is running and env/keys.env is configured")
        return None

@dataclass
class placements:
    pass

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

class es:
    @staticmethod
    def create_es_client():
        """Create and return Elasticsearch client"""
        load_dotenv(dotenv_path='env/keys.env')

        logger.info(f"Connecting to Elasticsearch at {os.getenv('ES_HOST')}...")

        client = Elasticsearch(
            hosts=[os.getenv('ES_HOST')], # type: ignore
            api_key=(os.getenv('ES_ID'), os.getenv('ES_SECRET')), # type: ignore
        )
        
        if client.ping():
            logger.info("Elasticsearch connection established successfully")
        else:
            logger.error("Failed to establish Elasticsearch connection")
            raise ValueError("Connection to Elasticsearch failed.")
        
        return client

    @staticmethod
    def refresh_index(client, index):
        """Refresh or recreate an Elasticsearch index"""
        if not client.indices.exists(index=index):
            print(f"    Creating index '{index}'...    ")
            resp = client.indices.create(index=index)
        else:
            print(f"    Index '{index}' already exists.    ")
            client.indices.delete(index=index)
            resp = client.indices.create(index=index)

    @staticmethod
    def create_tsds_index(client, index_name="bci_tsds"):
        """Create Time Series Data Stream (TSDS) index with proper mapping"""
        if client.indices.exists(index=index_name):
            return
            
        tsds_mapping = {
            "mappings": {
                "properties": {
                    "@timestamp": {"type": "date"},
                    "experiment_id": {"type": "keyword"},
                    "sequence_id": {"type": "long"},
                    "channel1": {"type": "float"},
                    "channel2": {"type": "float"},
                    "channel3": {"type": "float"},
                    "channel4": {"type": "float"},
                }
            }
        }
        
        client.indices.create(index=index_name, body=tsds_mapping)
        logger.info(f"Created TSDS index: {index_name}")

    @staticmethod
    def tsds_mapping(doc_id, ch1, ch2, ch3, ch4, timestamp=None, sequence_id=None):
        """Create TSDS document mapping"""
        return {
            "@timestamp": timestamp or datetime.now().isoformat(),
            "experiment_id": doc_id,
            "sequence_id": sequence_id,
            "channel1": ch1,
            "channel2": ch2,
            "channel3": ch3,
            "channel4": ch4,
        }

    @staticmethod
    def meta_mapping(exp):
        """Create experiment metadata mapping"""
        return exp.to_dict()

    @staticmethod
    def bulk_index_data(client, documents, index_name="bci_tsds"):
        """Bulk index documents to Elasticsearch"""
        try:
            if documents:
                actions = [{"_index": index_name, "_source": doc} for doc in documents]
                helpers.bulk(client, actions)
                logger.info(f"Indexed {len(documents)} documents to {index_name}")
                return True
        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")
            return False

    @staticmethod
    def index_experiment_metadata(client, experiment, meta_index="bci_experiments"):
        """Index experiment metadata and return the document ID"""
        try:
            resp = client.index(index=meta_index, document=experiment.to_es_doc())
            doc_id = resp['_id']
            logger.info(f"Indexed experiment metadata with ID: {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"Failed to index experiment metadata: {e}")
            return None

    @staticmethod
    def fetch_recent_data(client, index_name="bci_tsds", size=200, experiment_id=None):
        """Fetch recent documents from Elasticsearch for debugging"""
        try:
            if experiment_id:
                query = {
                    "query": {
                        "match": {
                            "experiment_id": experiment_id
                        }
                    },
                    "sort": [
                        {"@timestamp": {"order": "desc"}},
                        {"sequence_id": {"order": "desc"}}
                    ],
                    "size": size
                }
            else:
                query = {
                    "query": {
                        "match_all": {}
                    },
                    "sort": [
                        {"@timestamp": {"order": "desc"}},
                        {"sequence_id": {"order": "desc"}}
                    ],
                    "size": size
                }
            
            response = client.search(index=index_name, body=query)
            documents = response['hits']['hits']
            
            if not documents:
                logger.warning(f"No documents found in index: {index_name}")
                return []
            
            # Sort by timestamp ascending (oldest first)
            documents.reverse()
            return [doc['_source'] for doc in documents]
            
        except Exception as e:
            logger.error(f"Error fetching data from Elasticsearch: {e}")
            return []

    @staticmethod
    def fetch_incremental_data(client, index_name="bci_tsds", experiment_id=None, 
                              last_sequence_id=0, chunk_size=50):
        """Fetch only new data since last sequence ID for smoother streaming"""
        try:
            if experiment_id:
                query = {
                    "query": {
                        "bool": {
                            "must": [
                                {"match": {"experiment_id": experiment_id}},
                                {"range": {"sequence_id": {"gt": last_sequence_id}}}
                            ]
                        }
                    },
                    "sort": [{"sequence_id": {"order": "asc"}}],
                    "size": chunk_size
                }
            else:
                query = {
                    "query": {
                        "range": {"sequence_id": {"gt": last_sequence_id}}
                    },
                    "sort": [{"sequence_id": {"order": "asc"}}],
                    "size": chunk_size
                }
            
            response = client.search(index=index_name, body=query)
            documents = response['hits']['hits']
            
            return [doc['_source'] for doc in documents]
            
        except Exception as e:
            logger.error(f"Error fetching incremental data from Elasticsearch: {e}")
            return []

    @staticmethod
    def debug_plot_es_data(client, index_name="bci_tsds", size=200, experiment_id=None, realtime=False):
        """Debug function to plot data from Elasticsearch"""
        import matplotlib.pyplot as plt
        from collections import deque
        import matplotlib.animation as animation
        
        if realtime:
            return es._realtime_debug_plot(client, index_name, size, experiment_id)
        
        # Static plot
        documents = es.fetch_recent_data(client, index_name, size, experiment_id)
        
        if not documents:
            print("No data found to plot")
            return
        
        # Extract channel data
        timestamps = []
        ch1_data = []
        ch2_data = []
        ch3_data = []
        ch4_data = []
        
        for doc in documents:
            timestamps.append(doc.get('@timestamp', ''))
            ch1_data.append(doc.get('channel1', 0.0))
            ch2_data.append(doc.get('channel2', 0.0))
            ch3_data.append(doc.get('channel3', 0.0))
            ch4_data.append(doc.get('channel4', 0.0))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        x = range(len(ch1_data))
        
        ax.plot(x, ch1_data, label='Channel 1', color='blue', linewidth=1.5)
        ax.plot(x, ch2_data, label='Channel 2', color='red', linewidth=1.5)
        ax.plot(x, ch3_data, label='Channel 3', color='green', linewidth=1.5)
        ax.plot(x, ch4_data, label='Channel 4', color='orange', linewidth=1.5)
        
        ax.set_title(f'EEG Data from Elasticsearch ({index_name}) - {len(documents)} samples')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('EEG Value (µV)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Plotted {len(documents)} samples from experiment: {experiment_id or 'All'}")
        return fig, ax

    @staticmethod
    def _realtime_debug_plot(client, index_name="bci_tsds", size=200, experiment_id=None):
        """Real-time debug plotting from Elasticsearch"""
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from collections import deque
        
        # Data storage
        ch1_data = deque(maxlen=size)
        ch2_data = deque(maxlen=size)
        ch3_data = deque(maxlen=size)
        ch4_data = deque(maxlen=size)
        
        # Setup plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        lines = {}
        colors = ['blue', 'red', 'green', 'orange']
        channel_names = ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4']
        
        for i, (name, color) in enumerate(zip(channel_names, colors)):
            line, = ax.plot([], [], label=name, color=color, linewidth=1.5)
            lines[f'channel{i+1}'] = line
        
        ax.set_title(f'Real-time EEG Data from ES ({index_name})')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('EEG Value (µV)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, size)
        ax.set_ylim(-100, 100)
        
        def update_plot(frame):
            # Fetch latest data
            documents = es.fetch_recent_data(client, index_name, size, experiment_id)
            
            if not documents:
                return list(lines.values())
            
            # Clear and update data
            ch1_data.clear()
            ch2_data.clear()
            ch3_data.clear()
            ch4_data.clear()
            
            for doc in documents:
                ch1_data.append(doc.get('channel1', 0.0))
                ch2_data.append(doc.get('channel2', 0.0))
                ch3_data.append(doc.get('channel3', 0.0))
                ch4_data.append(doc.get('channel4', 0.0))
            
            # Update lines
            x_data = list(range(len(ch1_data)))
            lines['channel1'].set_data(x_data, list(ch1_data))
            lines['channel2'].set_data(x_data, list(ch2_data))
            lines['channel3'].set_data(x_data, list(ch3_data))
            lines['channel4'].set_data(x_data, list(ch4_data))
            
            # Auto-adjust y-limits
            all_data = list(ch1_data) + list(ch2_data) + list(ch3_data) + list(ch4_data)
            if all_data:
                y_min = min(all_data) * 1.1
                y_max = max(all_data) * 1.1
                if y_min != y_max:
                    ax.set_ylim(y_min, y_max)
            
            # Update title with sample count
            ax.set_title(f'Real-time EEG Data from ES ({index_name}) - {len(ch1_data)} samples')
            
            return list(lines.values())
        
        # Start animation
        ani = animation.FuncAnimation(fig, update_plot, interval=500, blit=False)
        
        print(f"Starting real-time debug plot from {index_name}")
        print("Press Ctrl+C or close window to stop")
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax, ani