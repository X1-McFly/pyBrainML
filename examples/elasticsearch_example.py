"""
Elasticsearch Integration Example for pyBrainML

This example demonstrates how to:
1. Set up Elasticsearch with proper indices and TSDS
2. Upload experiment data using the new architecture
3. Query and retrieve EEG data efficiently
4. Use real-time streaming for live data ingestion

Author: Martin McCorkle
Date: 2025-07-07
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv

import pybrainml as bml

def main():
    # Load environment variables
    dotenv_path = os.path.join(os.getcwd(), "env", 'keys.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
        print("Loading environment variables...")
    else:
        print("No environment file found - using default localhost settings")

    # Elasticsearch configuration
    ES_HOST = os.getenv("ES_HOST", "localhost")
    ES_PORT = int(os.getenv("ES_PORT", "9200"))
    ES_USERNAME = os.getenv("ES_USERNAME")
    ES_PASSWORD = os.getenv("ES_PASSWORD")
    
    print(f"Connecting to Elasticsearch at {ES_HOST}:{ES_PORT}")
    
    try:
        # Create Elasticsearch client
        es_client = bml.create_elasticsearch_client(
            host=ES_HOST,
            port=ES_PORT,
            username=ES_USERNAME,
            password=ES_PASSWORD,
            use_https=False
        )
        
        # Setup indices (experiments, frames, TSDS)
        es_client.setup_indices()
        
        # Load example experiment data
        data_file = os.path.join(os.getcwd(), "data", "test_session.json")
        experiment = bml.load_experiment_from_json(data_file)
        experiment_dict = experiment.to_dict()
        
        print(f"Loaded experiment with {len(experiment_dict.get('frames', []))} frames")
        
        # Define channel names (adjust based on your setup)
        channel_names = ["Fp1", "Fp2", "C3", "C4"]
        
        # Upload experiment to Elasticsearch
        print("\n=== Uploading Experiment to Elasticsearch ===")
        result = bml.upload_experiment_to_elasticsearch(
            experiment_dict, 
            es_client, 
            channel_names
        )
        
        experiment_id = str(result["experiment_id"])
        print(f"✓ Experiment uploaded with ID: {experiment_id}")
        print(f"✓ Frames uploaded: {result['frames_uploaded']}")
        
        # Demonstrate querying
        print("\n=== Querying Experiment Data ===")
        
        # Get experiment metadata
        metadata = es_client.get_experiment_metadata(experiment_id)
        print(f"✓ Retrieved metadata for subject: {metadata['subject_info']['subject_id'][:16]}...")
        
        # Get experiment frames
        frames = es_client.get_experiment_frames(experiment_id)
        print(f"✓ Retrieved {len(frames)} frames from frames index")
        
        # Query TSDS frames
        tsds_frames = es_client.search_tsds_frames(experiment_id)
        print(f"✓ Retrieved {len(tsds_frames)} frames from TSDS")
        
        # Demonstrate real-time streaming
        print("\n=== Real-Time Streaming Demo ===")
        
        # Create a new experiment for streaming
        streaming_exp = bml.create_experiment()
        streaming_exp.user_setup("Jane Doe", 28, "F")
        streaming_exp.hardware_setup(bml.ElectrodeType.DRY, bml.Boards.OpenBCI_Ganglion)
        
        # Create real-time streamer
        streamer = bml.create_realtime_streamer(
            es_client, 
            streaming_exp.to_dict(), 
            channel_names
        )
        
        # Start streaming
        streamer.start_streaming()
        
        # Simulate real-time data
        import time
        import random
        
        for i in range(10):
            # Simulate EEG frame
            eeg_data = [[random.uniform(-50, 50)] for _ in range(4)]
            frame = {
                "timestamp": datetime.now().isoformat(),
                "label": f"stream_frame_{i}",
                "eeg_data": eeg_data
            }
            
            streamer.add_frame(frame)
            time.sleep(0.1)  # 10 Hz simulation
        
        # Stop streaming
        streamer.stop_streaming()
        print("✓ Real-time streaming demo completed")
        
        # Query the streamed data
        streaming_experiment_id = streamer.experiment_id
        streamed_frames = es_client.search_tsds_frames(streaming_experiment_id)
        print(f"✓ Retrieved {len(streamed_frames)} streamed frames from TSDS")
        
        print("\n=== Example completed successfully! ===")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
