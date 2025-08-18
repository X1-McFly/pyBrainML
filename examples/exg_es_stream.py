"""
EXG Elasticsearch Real-Time Plotting Example

Author: Martin McCorkle
Date: 2025-08-06
Description:
    Demonstrates EEG data streaming to Elasticsearch with real-time plotting of ES data.
    Synthetic EEG data is uploaded to Elasticsearch and then pulled back in real-time
    for visualization. Simple, clean implementation modeled after exg_stream.py.

Features:
    - Streams synthetic EEG data to Elasticsearch
    - Real-time plotting of data pulled from Elasticsearch
    - Simple single-plot interface (no dual plots)
    - Round-trip data verification (device -> ES -> plot)

Dependencies:
    - pybrainml>=0.3.2
    - matplotlib
    - brainflow
    - elasticsearch
    - python-dotenv
"""

import time
from typing import Deque, List
import threading
from queue import Queue, Empty
import logging

import matplotlib.pyplot as plt

import pybrainml as bml
from pybrainml import ElectrodeType, Boards

# Silence Elasticsearch request logs
for name in ("elastic_transport", "elasticsearch", "urllib3"):
    logging.getLogger(name).setLevel(logging.WARNING)

def main():
    """Main function demonstrating Elasticsearch streaming with real-time plotting"""
    
    # Experiment setup
    port = "COM8"
    # port = None
    window_length = 200
    
    exp = bml.create_experiment()
    exp.user_setup(None, 35, "F")
    
    exp.hardware_setup(ElectrodeType.DRY, Boards.OpenBCI_Ganglion)
    # exp.hardware_setup(ElectrodeType.DRY, Boards.Synthetic)

    # Connect to board
    try:
        board_fd = bml.connect_board(exp, port) if port else bml.connect_board(exp)
    except ConnectionError as e:
        print(f"Failed to connect to board: {e}")
        print("Try using synthetic board for testing by commenting out the port and switching board type")
        return

    session = bml.exg_stream(
        board_fd, 
        length=window_length, 
        duration=None,
        use_elasticsearch=True, 
        experiment=exp,
        es_batch_size=25
    )
    
    # Start streaming
    session.start()
    time.sleep(1)
    
    plt.ion()
    fig, ax = plt.subplots()
    
    colors = ['blue', 'red', 'green', 'orange']
    channel_names = ['Chan 1', 'Chan 2', 'Chan 3', 'Chan 4']
    lines = []
    for i, (name, color) in enumerate(zip(channel_names, colors)):
        line, = ax.plot([], [], label=name, color=color, linewidth=1.5)
        lines.append(line)
    
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("EEG Value (µV)")
    ax.set_xlim(0, window_length)
    # ax.set_ylim(-25, 100)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Real-time EEG Data from Elasticsearch')

    # Shared data structures for threading
    data_queue = Queue()
    plot_buffer = []
    max_plot_points = window_length
    stop_threads = threading.Event()
    es_thread = None

    try:
        print("Real-time ES plotting started. Press Ctrl+C to stop...")
        
        def es_worker():
            """Background thread for fetching ES data"""
            last_sequence_id = 0
            
            while not stop_threads.is_set() and session.is_running():
                try:
                    es_client = session.get_es_client()
                    experiment_id = session.get_experiment_id()
                    
                    if es_client and experiment_id:
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
                            "size": 5
                        }
                        
                        response = es_client.search(index="bci_tsds", body=query)
                        new_documents = [doc['_source'] for doc in response['hits']['hits']]
                        
                        if new_documents:
                            plot_data = []
                            for doc in new_documents:
                                plot_data.append({
                                    'ch1': doc.get('channel1', 0.0),
                                    'ch2': doc.get('channel2', 0.0),
                                    'ch3': doc.get('channel3', 0.0),
                                    'ch4': doc.get('channel4', 0.0),
                                    'seq_id': doc.get('sequence_id', 0)
                                })
                                last_sequence_id = max(last_sequence_id, doc.get('sequence_id', 0))
                            
                            # Put new data in queue for plotting thread
                            data_queue.put(plot_data)
                except Exception as e:
                    print(f"ES worker error: {e}")
        
        # Start ES worker thread
        es_thread = threading.Thread(target=es_worker, daemon=True)
        es_thread.start()
        
        while session.is_running():
            try:
                # Get new data from ES worker (non-blocking)
                try:
                    while True:  # Process all available data
                        new_data = data_queue.get_nowait()
                        plot_buffer.extend(new_data)
                        data_queue.task_done()
                except Empty:
                    pass
                
                if len(plot_buffer) > max_plot_points:
                    plot_buffer = plot_buffer[-max_plot_points:]
                
                if plot_buffer:
                    es_ch1 = [point['ch1'] for point in plot_buffer]
                    es_ch2 = [point['ch2'] for point in plot_buffer]
                    es_ch3 = [point['ch3'] for point in plot_buffer]
                    es_ch4 = [point['ch4'] for point in plot_buffer]
                    
                    # Create x-axis data
                    x = list(range(len(es_ch1)))
                    
                    # Update plot lines with ES data
                    channel_traces = [es_ch1, es_ch2, es_ch3, es_ch4]
                    for line, vals in zip(lines, channel_traces):
                        line.set_data(x, vals)
                    
                    # Auto-adjust y-limits
                    all_vals = [v for vals in channel_traces for v in vals]
                    if all_vals:
                        ax.set_ylim(min(all_vals) * 1.1, max(all_vals) * 1.1)
                    
                    # Update x-limits and title
                    ax.set_xlim(0, len(es_ch1))
                    ax.set_title(f'Real-time EEG from ES - {len(plot_buffer)} samples')
                    
                    # Redraw plot
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                
            except Exception as e:
                print(f"Plot update error: {e}")
            
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        # Signal threads to stop
        stop_threads.set()
        
        print("Stopping session...")
        session.stop()
        
        # Wait for ES thread to finish
        if es_thread and es_thread.is_alive():
            es_thread.join(timeout=2)
        
        plt.ioff()
        plt.close()
        
        print("Session stopped. Data saved to Elasticsearch.")

    return

if __name__ == "__main__":
    
    main()
    
