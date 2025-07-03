from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv
import json
import os
import logging

# logging.basicConfig(level=logging.DEBUG)

# CONFIG
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

print(f"ES_HOST: {ES_HOST}")
print(f"ES_ID: {ES_ID}")
print(f"ES_SECRET: {ES_SECRET}")
INDEX_NAME = "biocom"

def init():
    try:
        es = Elasticsearch([ES_HOST], api_key=(ES_ID, ES_SECRET), verify_certs=True) # type: ignore
        if not es.ping():
            raise ValueError("Ping failed: could not connect to cluster.")
        print("Connected to Elasticsearch successfully!")
    except Exception as e:
        print(f"[!] Elasticsearch connection failed:\n{e}")
        raise


# def load_json_blocks(filepath):
#     with open(filepath, 'r', encoding='utf-8') as f:
#         print(f"Loading JSON blocks from {filepath}...")
#         print(f"File size: {os.path.getsize(filepath)} bytes")
#         data = json.load(f)
#         print(f"Loaded {len(data)} blocks.")
#         return data

# def bulk_upload(es, json_file, index):
#     # pass
#     blocks = load_json_blocks(json_file)
#     actions = []
#     for doc in blocks:
#         if isinstance(doc, dict):
#             actions.append({"_index": index, "_source": doc})
#         else:
#             actions.append({"_index": index, "_source": doc})

#     if actions:
#         try:
#             helpers.bulk(es, actions)
#             print(f"Uploaded {len(actions)} documents from {json_file}")
#         except helpers.BulkIndexError as e:
#             print(f"BulkIndexError: {len(e.errors)} documents failed.")
#             for error in e.errors:
#                 print(json.dumps(error, indent=2))

if __name__ == "__main__":
    es = init()
    # bulk_upload(es, DATA_BLOCK_FILE, INDEX_NAME)
    print("Done.")
