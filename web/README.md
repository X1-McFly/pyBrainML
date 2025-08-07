# EEG Elasticsearch Real-Time Viewer

A React web application for real-time visualization of EEG data stored in Elasticsearch.

## Features

- Real-time EEG data streaming from Elasticsearch
- Interactive Chart.js visualization with 4 EEG channels
- Configurable Elasticsearch connection
- Automatic data buffering and performance optimization
- Responsive design for desktop and mobile

## Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- Elasticsearch cluster with EEG data
- EEG data indexed with the following structure:
  ```json
  {
    "timestamp": "2024-01-01T00:00:00.000Z",
    "experiment_id": "exp_001",
    "ch1": 0.001,
    "ch2": 0.002,
    "ch3": 0.003,
    "ch4": 0.004
  }
  ```

## Installation

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

3. Open your browser and navigate to `http://localhost:3000`

## Usage

1. **Connect to Elasticsearch:**
   - Enter your Elasticsearch node URL (e.g., `http://localhost:9200`)
   - Provide username and password
   - Specify the index name containing EEG data
   - Click "Connect"

2. **Start Streaming:**
   - Enter an experiment ID that exists in your Elasticsearch index
   - Click "Start Streaming"
   - The chart will begin displaying real-time EEG data

3. **View Data:**
   - The chart displays 4 EEG channels in different colors
   - Data is automatically updated at 20 FPS
   - The chart maintains the last 2000 data points for optimal performance
   - Hover over the chart to see detailed values

## Configuration

### Elasticsearch Index Mapping

Your Elasticsearch index should have the following mapping for optimal performance:

```json
{
  "mappings": {
    "properties": {
      "timestamp": {
        "type": "date",
        "format": "strict_date_optional_time||epoch_millis"
      },
      "experiment_id": {
        "type": "keyword"
      },
      "ch1": { "type": "float" },
      "ch2": { "type": "float" },
      "ch3": { "type": "float" },
      "ch4": { "type": "float" }
    }
  }
}
```

### Performance Settings

The application is optimized for real-time streaming:
- Fetch interval: 50ms (20 FPS)
- Maximum data points: 2000
- Chart animations disabled for performance
- Incremental data fetching using Elasticsearch `_seq_no`

## Architecture

- **React Hooks:** Custom `useElasticsearchData` hook manages data fetching
- **Elasticsearch Service:** Dedicated service class for ES operations
- **Chart.js:** High-performance charting with Chart.js and react-chartjs-2
- **Real-time Updates:** Continuous data polling with automatic buffering

## Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

### Project Structure

```
src/
├── components/
│   ├── ElasticsearchConnection.jsx  # Connection management
│   └── EEGChart.jsx                 # Chart visualization
├── hooks/
│   └── useElasticsearchData.js      # Data fetching logic
├── services/
│   └── elasticsearchService.js      # Elasticsearch client
├── App.jsx                          # Main application
└── main.jsx                         # Application entry point
```

## Troubleshooting

### Connection Issues
- Verify Elasticsearch is running and accessible
- Check username/password credentials
- Ensure CORS is properly configured on Elasticsearch
- For local development, the app disables SSL verification

### Performance Issues
- Reduce the fetch interval if experiencing lag
- Decrease the maximum data points buffer
- Check Elasticsearch cluster performance
- Monitor browser memory usage

### Data Issues
- Verify the experiment ID exists in the index
- Check the index mapping matches expected structure
- Ensure data is being continuously indexed
- Use Elasticsearch Dev Tools to verify queries

## Dependencies

- React 18
- Chart.js 4.4
- @elasticsearch/elasticsearch 8.12
- Vite (build tool)

## License

This project is part of the pyBrainML toolkit for EEG data analysis.
