# pyBrainML - Elasticsearch Integration Guide

## Overview

pyBrainML now includes comprehensive Elasticsearch integration following a Time Series Data Stream (TSDS) architecture for optimal EEG data storage, querying, and real-time streaming.

## Architecture

### Index Strategy

| Index/Stream     | Purpose                              | Key Fields                    |
|------------------|--------------------------------------|-------------------------------|
| `experiments`    | Experiment metadata (one per run)   | `experiment_id`, subject info |
| `frames`         | Individual EEG frames               | `experiment_id`, `frame_id`   |
| `frames-tsds`    | Real-time streaming (TSDS)          | `@timestamp`, channel data    |

### Data Flow

1. **Metadata Upload**: Experiment information stored in `experiments` index
2. **Bulk Frame Upload**: Historical EEG data stored in `frames` index  
3. **Real-time Streaming**: Live data ingested to `frames-tsds` TSDS
4. **Querying**: Efficient retrieval from appropriate index/stream

## Quick Start

### 1. Setup Elasticsearch Client

```python
import pybrainml as bml

# Create client
es_client = bml.create_elasticsearch_client(
    host="localhost",
    port=9200,
    username="your_username",  # Optional
    password="your_password",  # Optional
    use_https=False
)

# Setup indices and TSDS
es_client.setup_indices()
channel_names = ["Fp1", "Fp2", "C3", "C4"]
es_client.setup_tsds(channel_names)
```

### 2. Upload Experiment Data

```python
# Load experiment
experiment = bml.load_experiment_from_json("data/session.json")
experiment_dict = experiment.to_dict()

# Upload to Elasticsearch
result = bml.upload_experiment_to_elasticsearch(
    experiment_dict,
    es_client,
    channel_names
)

experiment_id = result["experiment_id"]
print(f"Uploaded experiment: {experiment_id}")
```

### 3. Query Data

```python
# Get experiment metadata
metadata = es_client.get_experiment_metadata(experiment_id)

# Get all frames
frames = es_client.get_experiment_frames(experiment_id)

# Query TSDS with time filtering
from datetime import datetime, timedelta
end_time = datetime.now()
start_time = end_time - timedelta(hours=1)

tsds_frames = es_client.search_tsds_frames(
    experiment_id,
    start_time=start_time.isoformat(),
    end_time=end_time.isoformat()
)
```

### 4. Real-time Streaming

```python
# Create real-time streamer
streamer = bml.create_realtime_streamer(
    es_client,
    experiment_dict,
    channel_names
)

# Start streaming
streamer.start_streaming()

# Add frames (e.g., from live EEG device)
for eeg_sample in live_eeg_stream():
    frame = {
        "timestamp": datetime.now().isoformat(),
        "label": "live_data",
        "eeg_data": eeg_sample  # [[ch1_val], [ch2_val], ...]
    }
    streamer.add_frame(frame)

# Stop streaming
streamer.stop_streaming()
```

## API Reference

### ElasticsearchClient

#### Core Methods

- `setup_indices()`: Create experiments and frames indices
- `setup_tsds(channel_names)`: Setup TSDS for real-time streaming
- `upload_experiment_metadata(experiment_dict)`: Upload metadata, returns experiment_id
- `upload_frames_bulk(experiment_id, frames, channel_names)`: Bulk upload frames
- `upload_frame_to_tsds(experiment_id, frame, channel_names)`: Stream single frame

#### Query Methods

- `get_experiment_metadata(experiment_id)`: Get experiment metadata
- `get_experiment_frames(experiment_id, start_time, end_time)`: Query frames
- `search_tsds_frames(experiment_id, start_time, end_time)`: Query TSDS

### RealTimeEEGStreamer

- `start_streaming()`: Begin real-time streaming
- `add_frame(frame_data)`: Add frame to streaming buffer
- `stop_streaming()`: Stop streaming and flush buffer

### Utility Functions

- `create_elasticsearch_client()`: Create ES client instance
- `upload_experiment_to_elasticsearch()`: Complete experiment upload
- `create_realtime_streamer()`: Create real-time streamer

## Data Formats

### Experiment Metadata

```json
{
  "experiment_id": "uuid",
  "version": "0.3.2",
  "created_at": "2025-07-07T12:00:00",
  "subject_info": {
    "subject_id": "hashed_name",
    "age": 35,
    "sex": "F"
  },
  "hardware_info": {
    "electrode_type": "DRY",
    "board": "OpenBCI_Ganglion",
    "channels": 4,
    "sampling_rate": 200
  }
}
```

### Frame Document

```json
{
  "experiment_id": "uuid",
  "frame_id": "uuid", 
  "timestamp": "2025-07-07T12:00:01",
  "label": "eyes_closed",
  "eeg": {
    "Fp1": 8.16,
    "Fp2": -1.42,
    "C3": 6.34,
    "C4": 2.07
  }
}
```

### TSDS Document

```json
{
  "@timestamp": "2025-07-07T12:00:01",
  "experiment_id": "uuid",
  "frame_id": "uuid",
  "label": "eyes_closed", 
  "eeg.Fp1": 8.16,
  "eeg.Fp2": -1.42,
  "eeg.C3": 6.34,
  "eeg.C4": 2.07
}
```

## Configuration

### Environment Variables

Create `env/keys.env`:

```bash
ES_HOST=your-elasticsearch-host
ES_USERNAME=your_username
ES_PASSWORD=your_password
ES_PORT=9200
```

### Index Templates

The library automatically creates optimized index templates:

- **Experiments**: Keyword-optimized for metadata queries
- **Frames**: Balanced for frame-level search and aggregation  
- **TSDS**: Time-series optimized with routing and sorting

## Examples

See the `examples/` directory:

- `elasticsearch_example.py`: Complete integration example
- `realtime_es_streaming.py`: Real-time streaming with visualization
- `es_test.py`: Updated test script with new architecture

## Performance Tips

1. **Batch Size**: Use bulk uploads for historical data
2. **Channel Names**: Use meaningful channel names (e.g., "Fp1" vs "Ch1")  
3. **Time Filtering**: Always use time ranges for large datasets
4. **Index Lifecycle**: Configure ILM policies for data retention
5. **Buffer Size**: Adjust streaming buffer size based on data rate

## Troubleshooting

### Connection Issues
- Verify Elasticsearch is running and accessible
- Check authentication credentials
- Ensure network connectivity

### Performance Issues  
- Monitor Elasticsearch cluster health
- Adjust bulk upload batch sizes
- Consider index sharding for large datasets

### Data Issues
- Validate channel names match hardware setup
- Check timestamp formats (ISO 8601 required)
- Verify EEG data array structure

## Migration from Legacy Code

To migrate from the old upload functions:

```python
# Old way
upload_experiment_to_es(exp_dict, "index_name", "host", 9200, "user", "pass")

# New way  
es_client = bml.create_elasticsearch_client("host", 9200, "user", "pass")
result = bml.upload_experiment_to_elasticsearch(exp_dict, es_client, channel_names)
```

The new architecture provides:
- Better performance through proper indexing
- Real-time streaming capabilities  
- Standardized data formats
- Comprehensive querying options
- Future-proof scalability
