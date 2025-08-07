import { useState, useEffect, useRef, useCallback } from 'react';
import { ElasticsearchService } from '../services/elasticsearchService';

const useElasticsearchData = (connectionConfig, experimentId) => {
  const [data, setData] = useState({ timestamps: [], channels: [[], [], [], []] });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isFetching, setIsFetching] = useState(false);
  
  const esServiceRef = useRef(null);
  const intervalRef = useRef(null);
  const sequenceIdRef = useRef(0);

  // Initialize Elasticsearch service when config changes
  useEffect(() => {
    if (connectionConfig) {
      try {
        esServiceRef.current = new ElasticsearchService(connectionConfig);
        setError(null);
      } catch (err) {
        setError(`Failed to initialize Elasticsearch service: ${err.message}`);
      }
    }
  }, [connectionConfig]);

  // Fetch incremental data
  const fetchIncrementalData = useCallback(async () => {
    if (!esServiceRef.current || !experimentId) return;

    try {
      setIsLoading(true);
      
      const result = await esServiceRef.current.fetchIncrementalData(
        experimentId,
        sequenceIdRef.current
      );

      if (result && result.newData.length > 0) {
        // Update sequence ID for next fetch
        sequenceIdRef.current = result.lastSequenceId;

        // Process new data
        const newTimestamps = result.newData.map(doc => doc.timestamp);
        const newChannelData = [[], [], [], []];
        
        result.newData.forEach(doc => {
          if (doc.ch1 !== undefined) newChannelData[0].push(doc.ch1);
          if (doc.ch2 !== undefined) newChannelData[1].push(doc.ch2);
          if (doc.ch3 !== undefined) newChannelData[2].push(doc.ch3);
          if (doc.ch4 !== undefined) newChannelData[3].push(doc.ch4);
        });

        // Update state with new data
        setData(prevData => {
          const maxPoints = 2000; // Keep last 2000 points for performance
          
          const updatedTimestamps = [...prevData.timestamps, ...newTimestamps];
          const updatedChannels = prevData.channels.map((channel, index) => 
            [...channel, ...newChannelData[index]]
          );

          // Trim if too many points
          if (updatedTimestamps.length > maxPoints) {
            const trimAmount = updatedTimestamps.length - maxPoints;
            return {
              timestamps: updatedTimestamps.slice(trimAmount),
              channels: updatedChannels.map(channel => channel.slice(trimAmount))
            };
          }

          return {
            timestamps: updatedTimestamps,
            channels: updatedChannels
          };
        });
      }
      
      setError(null);
    } catch (err) {
      console.error('Error fetching data:', err);
      setError(`Failed to fetch data: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  }, [experimentId]);

  // Start fetching data
  const startFetching = useCallback(() => {
    if (isFetching) return;
    
    setIsFetching(true);
    setError(null);
    sequenceIdRef.current = 0; // Reset sequence ID
    setData({ timestamps: [], channels: [[], [], [], []] }); // Reset data

    // Fetch immediately
    fetchIncrementalData();

    // Set up interval for continuous fetching (20 FPS = 50ms)
    intervalRef.current = setInterval(() => {
      fetchIncrementalData();
    }, 50);
  }, [fetchIncrementalData, isFetching]);

  // Stop fetching data
  const stopFetching = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setIsFetching(false);
    setIsLoading(false);
  }, []);

  // Cleanup on unmount or when experiment changes
  useEffect(() => {
    return () => {
      stopFetching();
    };
  }, [stopFetching]);

  // Stop fetching when experiment changes
  useEffect(() => {
    if (!experimentId) {
      stopFetching();
    }
  }, [experimentId, stopFetching]);

  return {
    data,
    isLoading,
    error,
    isFetching,
    startFetching,
    stopFetching
  };
};

export default useElasticsearchData;
