import React, { useState } from 'react';
import './ElasticsearchConnection.css';

const ElasticsearchConnection = ({ 
  onConnect, 
  onDisconnect, 
  isConnected, 
  onStartStreaming, 
  onStopStreaming, 
  isStreaming 
}) => {
  const [config, setConfig] = useState({
    node: 'http://localhost:9200',
    username: 'elastic',
    password: '',
    index: 'eeg_data'
  });
  const [experimentId, setExperimentId] = useState('');
  const [isConnecting, setIsConnecting] = useState(false);

  const handleInputChange = (field, value) => {
    setConfig(prev => ({ ...prev, [field]: value }));
  };

  const handleConnect = async () => {
    setIsConnecting(true);
    try {
      await onConnect(config);
    } catch (error) {
      console.error('Connection failed:', error);
    } finally {
      setIsConnecting(false);
    }
  };

  const handleStartStreaming = () => {
    if (experimentId.trim()) {
      onStartStreaming(experimentId.trim());
    }
  };

  if (isConnected) {
    return (
      <div className="es-connection connected">
        <div className="connection-status">
          <div className="status-indicator connected"></div>
          <span>Connected to {config.node}</span>
          <button className="disconnect-btn" onClick={onDisconnect}>
            Disconnect
          </button>
        </div>
        
        <div className="streaming-controls">
          <div className="input-group">
            <label htmlFor="experiment-id">Experiment ID:</label>
            <input
              id="experiment-id"
              type="text"
              value={experimentId}
              onChange={(e) => setExperimentId(e.target.value)}
              placeholder="Enter experiment ID"
              disabled={isStreaming}
            />
          </div>
          
          <div className="streaming-buttons">
            {!isStreaming ? (
              <button 
                className="start-streaming-btn"
                onClick={handleStartStreaming}
                disabled={!experimentId.trim()}
              >
                Start Streaming
              </button>
            ) : (
              <button className="stop-streaming-btn" onClick={onStopStreaming}>
                Stop Streaming
              </button>
            )}
          </div>
          
          {isStreaming && (
            <div className="streaming-status">
              <div className="status-indicator streaming"></div>
              <span>Streaming experiment: {experimentId}</span>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="es-connection">
      <h3>Elasticsearch Connection</h3>
      
      <div className="connection-form">
        <div className="input-group">
          <label htmlFor="es-node">Elasticsearch Node:</label>
          <input
            id="es-node"
            type="text"
            value={config.node}
            onChange={(e) => handleInputChange('node', e.target.value)}
            placeholder="http://localhost:9200"
          />
        </div>
        
        <div className="input-group">
          <label htmlFor="es-username">Username:</label>
          <input
            id="es-username"
            type="text"
            value={config.username}
            onChange={(e) => handleInputChange('username', e.target.value)}
            placeholder="elastic"
          />
        </div>
        
        <div className="input-group">
          <label htmlFor="es-password">Password:</label>
          <input
            id="es-password"
            type="password"
            value={config.password}
            onChange={(e) => handleInputChange('password', e.target.value)}
            placeholder="Enter password"
          />
        </div>
        
        <div className="input-group">
          <label htmlFor="es-index">Index Name:</label>
          <input
            id="es-index"
            type="text"
            value={config.index}
            onChange={(e) => handleInputChange('index', e.target.value)}
            placeholder="eeg_data"
          />
        </div>
        
        <button 
          className="connect-btn"
          onClick={handleConnect}
          disabled={isConnecting || !config.node || !config.username || !config.password}
        >
          {isConnecting ? 'Connecting...' : 'Connect'}
        </button>
      </div>
    </div>
  );
};

export default ElasticsearchConnection;
