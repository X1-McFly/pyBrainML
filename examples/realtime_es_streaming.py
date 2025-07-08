"""
Real-Time EEG Streaming with Elasticsearch Integration

This example shows how to:
1. Stream EEG data in real-time to Elasticsearch TSDS
2. Visualize the data while streaming
3. Query live data from Elasticsearch

Author: Martin McCorkle  
Date: 2025-07-07
"""

import time
from typing import Deque, List
import matplotlib.pyplot as plt
from collections import deque

import pybrainml as bml
from pybrainml import ElectrodeType, Boards

def main():
    # Experiment setup
    port = "COM8"
    data_dir = "data"
    window_length = 200
    use_elasticsearch = True  # Set to False to disable ES integration

    exp = bml.create_experiment()
    exp.user_setup("John Doe", 35, "F")
    exp.hardware_setup(ElectrodeType.DRY, Boards.OpenBCI_Ganglion)

    # Elasticsearch setup (optional)
    es_streamer = None
    if use_elasticsearch:
        try:
            # Create ES client (adjust connection parameters as needed)
            es_client = bml.create_elasticsearch_client(
                host="localhost",
                port=9200,
                username=None,  # Set if required
                password=None,  # Set if required
                use_https=False
            )
            
            # Create real-time streamer
            channel_names = ["Ch1", "Ch2", "Ch3", "Ch4"]
            es_streamer = bml.create_realtime_streamer(
                es_client, 
                exp.to_dict(), 
                channel_names
            )
            es_streamer.start_streaming()
            print("✓ Elasticsearch real-time streaming enabled")
            
        except Exception as e:
            print(f"⚠️  Elasticsearch streaming disabled: {e}")
            es_streamer = None

    # Connect to board and prepare streaming session
    try:
        board_fd = bml.connect_board(port, Boards.OpenBCI_Ganglion)
        session = bml.exg_stream(board_fd, length=window_length)
    except Exception as e:
        print(f"Could not connect to board: {e}")
        print("Running in simulation mode...")
        board_fd = None
        session = None

    # Prepare real-time plot
    plt.ion()
    fig, ax = plt.subplots()
    
    if session:
        num_ch = len(session.eeg_channels())
    else:
        num_ch = 4  # Simulation mode
        
    lines = [ax.plot([], [], label=f"Chan {i+1}")[0] for i in range(num_ch)]
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("EXG Value")
    ax.set_xlim(0, window_length)
    ax.legend()

    # Start experiment
    if session:
        session.start()

    frame_count = 0
    
    try:
        while True:
            time.sleep(0.01)
            
            if session:
                # Real board data
                buf: Deque[List[float | str]] = session.get_buffer()
                if not buf:
                    continue
                    
                x = list(range(len(buf)))
                channel_traces: List[List[float]] = [
                    [float(sample[i+1]) for sample in buf] for i in range(num_ch)
                ]
            else:
                # Simulation mode
                import random
                buf_size = min(window_length, 50)
                x = list(range(buf_size))
                channel_traces = [
                    [random.uniform(-50, 50) for _ in range(buf_size)] 
                    for _ in range(num_ch)
                ]

            # Update plot
            for line, vals in zip(lines, channel_traces):
                line.set_data(x, vals)
                
            all_vals = [v for vals in channel_traces for v in vals]
            if all_vals:
                ax.set_ylim(min(all_vals) * 1.1, max(all_vals) * 1.1)
            
            fig.canvas.draw()
            fig.canvas.flush_events()

            # Stream to Elasticsearch every 10 frames
            if es_streamer and frame_count % 10 == 0:
                try:
                    # Convert to pybrainml frame format
                    eeg_data = [[trace[0]] if trace else [0.0] for trace in channel_traces]
                    frame = {
                        "timestamp": time.time(),
                        "label": f"realtime_frame_{frame_count}",
                        "eeg_data": eeg_data
                    }
                    es_streamer.add_frame(frame)
                except Exception as e:
                    print(f"ES streaming error: {e}")

            frame_count += 1

            if session and not session.is_running():
                break
                
            # Exit after 1000 frames in simulation mode
            if not session and frame_count > 1000:
                break

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        if session:
            print("Stopping session...")
            session.stop()
            bml.export_experiment(session, exp, data_dir)
        
        if es_streamer:
            es_streamer.stop_streaming()
            print(f"✓ Streamed {frame_count} frames to Elasticsearch")
            
            # Demonstrate querying the live data
            try:
                experiment_id = es_streamer.experiment_id
                tsds_frames = es_streamer.es_client.search_tsds_frames(experiment_id)
                print(f"✓ Query result: {len(tsds_frames)} frames in TSDS")
            except Exception as e:
                print(f"Query error: {e}")
        
        plt.ioff()
        plt.close()

if __name__ == "__main__":
    main()
