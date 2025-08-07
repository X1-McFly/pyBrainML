from datetime import datetime
import math
import random
import uuid
from elasticsearch import Elasticsearch, helpers  
import dotenv
import os
import time
import pybrainml as bml
from pybrainml import ElectrodeType, Boards
from typing import List, Dict, Deque, Set
import socket
import threading
from queue import Queue, Empty
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s.%(msecs)03d] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

meta_doc = "bci_experiments"
tsds = "bci_tsds"

uuid = str(uuid.uuid4())

def create_es_client():
    
    dotenv.load_dotenv(dotenv_path='env/keys.env')

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

def refresh_index(client, index):
    if not client.indices.exists(index=index):
        print(f"    Creating index '{index}'...    ")
        resp = client.indices.create(index=index)
    else:
        print(f"    Index '{index}' already exists.    ")
        client.indices.delete(index=index)
        resp = client.indices.create(index=index)

def tsds_mapping(doc_id, ch1, ch2, ch3, ch4, timestamp=None, sequence_id=None):
    return {
        "@timestamp": timestamp or datetime.now().isoformat(),
        "experiment_id": doc_id,
        "sequence_id": sequence_id,
        "channel1": ch1,
        "channel2": ch2,
        "channel3": ch3,
        "channel4": ch4,
    }

def meta_mapping(exp):
    return exp.to_dict()

if __name__ == "__main__":

    data_dir = "data"
    window_length = 200
    exp = bml.create_experiment()
    exp.user_setup(None, 35, "F")
    exp.hardware_setup(ElectrodeType.DRY, Boards.Synthetic)

    # For synthetic board, no port needed
    board_fd = bml.connect_board(exp)
    session = bml.exg_stream(board_fd, length=window_length, duration=None)

    client = create_es_client()
    if not client.ping():
        raise ValueError("Connection to Elasticsearch failed.")

    # refresh_index(client, tsds)
    # refresh_index(client, meta_doc)

    print("    Elasticsearch client created successfully.   ")

    print(meta_mapping(exp))

    resp = client.index(index=meta_doc, document=exp.to_es_doc())
    doc_id = resp['_id']
    print(f"    Last indexed document ID: {doc_id}   ")

    session.start()

    buff = []
    tic = datetime.now()
    previous_buffer_size = 0
    try:
        while session.is_running():
            # Get current buffer
            buf = session.get_buffer()
            
            # Only process new data points since last iteration
            current_buffer_size = len(buf)
            if current_buffer_size > previous_buffer_size:
                # Convert deque to list and get only new items
                buf_list = list(buf)
                new_data_points = buf_list[previous_buffer_size:]
                
                # Process each new data point
                for data_point in new_data_points:
                    if len(data_point) >= 5:  # timestamp + 4 channels
                        # Extract channels (skip timestamp at index 0)
                        ch1, ch2, ch3, ch4 = data_point[1:5]
                        timestamp = datetime.now().isoformat()
                        fps = 1 / (datetime.now() - tic).total_seconds() if (datetime.now() - tic).total_seconds() > 0 else 0
                        print(f"    Received data: {ch1}, {ch2}, {ch3}, {ch4} at {timestamp}, FPS: {fps:.2f}   ")
                        tic = datetime.now()
                        
                        # Create document for Elasticsearch
                        doc = tsds_mapping(
                            doc_id=doc_id,
                            ch1=ch1,
                            ch2=ch2, 
                            ch3=ch3,
                            ch4=ch4,
                            timestamp=timestamp,
                            sequence_id=len(buff)
                        )
                        buff.append({"_index": tsds, "_source": doc})
                
                # Update the previous buffer size
                previous_buffer_size = current_buffer_size
                
                # Bulk insert when buffer reaches certain size
                if len(buff) >= 100:
                    helpers.bulk(client, buff)
                    buff.clear()
            
            time.sleep(0.01)  # Small delay to prevent excessive CPU usage
    except KeyboardInterrupt:
        pass

    finally:
            print("Stopping session...")
            session.stop()
            count = client.count(index=tsds)['count']
            print(f"    Number of documents in '{tsds}': {count}   ")

    toc = datetime.now()
    elapsed_seconds = (toc - tic).total_seconds()
    upload_speed = count / elapsed_seconds if elapsed_seconds > 0 else 0
    print(f"    Upload speed: {upload_speed:.2f} docs/sec   ")
    print(f"    Bulk insert took {toc - tic} seconds.   ")