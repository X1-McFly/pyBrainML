from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv
import json
import os
import logging

# Import the new pybrainml elasticsearch functionality
import pybrainml as bml

# logging.basicConfig(level=logging.DEBUG)

DATA_BLOCK_FILE = os.path.join(os.getcwd(),"data", "test_session.json")

# CONFIG
dotenv_path = os.path.join(os.getcwd(), "env", 'keys.env')
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

print(f"ES_HOST: {ES_HOST}")
print(f"ES_ID: {ES_ID}")
print(f"ES_SECRET: {ES_SECRET}")

def main():
    """
    Updated main function using the new Elasticsearch architecture.
    """
    try:
        # Create Elasticsearch client using the new pybrainml functionality
        host_clean = (ES_HOST or "localhost").replace("https://", "").replace("http://", "")
        es_client = bml.create_elasticsearch_client(
            host=host_clean,
            port=9200,  # Adjust if different
            username=ES_ID,
            password=ES_SECRET,
            use_https=True
        )
        
        # Load experiment data
        exp = bml.load_experiment_from_json(DATA_BLOCK_FILE)
        exp_dict = exp.to_dict()
        print(f"Loaded experiment with {len(exp_dict.get('frames', []))} frames")
        
        # Define channel names based on your hardware
        channel_names = ["Ch1", "Ch2", "Ch3", "Ch4"]  # Adjust based on your setup
        
        # Upload experiment using the new architecture
        result = bml.upload_experiment_to_elasticsearch(
            exp_dict, 
            es_client, 
            channel_names
        )
        
        experiment_id = str(result["experiment_id"])
        print(f"✓ Experiment uploaded with ID: {experiment_id}")
        print(f"✓ Frames uploaded: {result['frames_uploaded']}")
        
        # Query the uploaded data
        metadata = es_client.get_experiment_metadata(experiment_id)
        print(f"✓ Retrieved metadata for experiment")
        
        frames = es_client.get_experiment_frames(experiment_id)
        print(f"✓ Retrieved {len(frames)} frames from frames index")
        
        # Query TSDS
        tsds_frames = es_client.search_tsds_frames(experiment_id)
        print(f"✓ Retrieved {len(tsds_frames)} frames from TSDS")
        
        print("Upload and query completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()