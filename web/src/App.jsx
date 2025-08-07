import React, { useState, useEffect } from 'react';
import ElasticsearchConnection from './components/ElasticsearchConnection';
import EEGChart from './components/EEGChart';
import useElasticsearchData from './hooks/useElasticsearchData';
import './App.css';

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionConfig, setConnectionConfig] = useState(null);
  const [experimentId, setExperimentId] = useState('');
  
  const { data, isLoading, error, startFetching, stopFetching } = useElasticsearchData(
    connectionConfig,
    experimentId
  );

  const handleConnect = (config) => {
    setConnectionConfig(config);
    setIsConnected(true);
  };

  const handleDisconnect = () => {
    stopFetching();
    setIsConnected(false);
    setConnectionConfig(null);
  };

  const handleStartStreaming = (expId) => {
    setExperimentId(expId);
    startFetching();
  };

  const handleStopStreaming = () => {
    stopFetching();
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>EEG Elasticsearch Real-Time Viewer</h1>
        <p>Connect to Elasticsearch and visualize EEG data in real-time</p>
      </header>

      <main className="App-main">
        <div className="connection-panel">
          <ElasticsearchConnection
            onConnect={handleConnect}
            onDisconnect={handleDisconnect}
            isConnected={isConnected}
            onStartStreaming={handleStartStreaming}
            onStopStreaming={handleStopStreaming}
            isStreaming={!!experimentId}
          />
        </div>

        {error && (
          <div className="error-panel">
            <h3>Error:</h3>
            <p>{error}</p>
          </div>
        )}

        {isConnected && (
          <div className="chart-panel">
            <EEGChart 
              data={data} 
              isLoading={isLoading}
              experimentId={experimentId}
            />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
