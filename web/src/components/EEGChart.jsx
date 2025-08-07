import React, { useRef, useEffect } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import './EEGChart.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const EEGChart = ({ data, isLoading, experimentId }) => {
  const chartRef = useRef(null);

  const channelColors = [
    '#FF6B6B', // Red
    '#4ECDC4', // Teal
    '#45B7D1', // Blue
    '#96CEB4'  // Green
  ];

  // Prepare chart data
  const chartData = {
    labels: data?.timestamps || [],
    datasets: data?.channels?.map((channelData, index) => ({
      label: `Channel ${index + 1}`,
      data: channelData,
      borderColor: channelColors[index % channelColors.length],
      backgroundColor: channelColors[index % channelColors.length] + '20',
      borderWidth: 1.5,
      fill: false,
      pointRadius: 0,
      pointHoverRadius: 3,
      tension: 0.1,
    })) || []
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 0 // Disable animations for real-time performance
    },
    interaction: {
      intersect: false,
      mode: 'index'
    },
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: experimentId ? `EEG Data - Experiment: ${experimentId}` : 'EEG Data Stream'
      },
      tooltip: {
        position: 'nearest',
        callbacks: {
          title: (tooltipItems) => {
            if (tooltipItems[0]) {
              return `Time: ${tooltipItems[0].label}`;
            }
            return '';
          },
          label: (context) => {
            return `${context.dataset.label}: ${context.parsed.y.toFixed(4)}`;
          }
        }
      }
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Time'
        },
        ticks: {
          maxTicksLimit: 10,
          callback: function(value, index, values) {
            // Show only every nth timestamp to avoid crowding
            const timestamp = this.getLabelForValue(value);
            return timestamp ? new Date(timestamp).toLocaleTimeString() : '';
          }
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Amplitude (µV)'
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        }
      }
    },
    elements: {
      line: {
        tension: 0.1
      }
    }
  };

  // Auto-scroll to latest data
  useEffect(() => {
    if (chartRef.current && data?.timestamps?.length > 0) {
      const chart = chartRef.current;
      // Keep only the last 1000 points for performance
      const maxPoints = 1000;
      if (data.timestamps.length > maxPoints) {
        const startIndex = data.timestamps.length - maxPoints;
        chart.data.labels = data.timestamps.slice(startIndex);
        chart.data.datasets.forEach((dataset, index) => {
          dataset.data = data.channels[index]?.slice(startIndex) || [];
        });
        chart.update('none'); // Update without animation
      }
    }
  }, [data]);

  if (!experimentId) {
    return (
      <div className="eeg-chart-container">
        <div className="no-data">
          <h3>No Experiment Selected</h3>
          <p>Please connect to Elasticsearch and start streaming an experiment to view EEG data.</p>
        </div>
      </div>
    );
  }

  if (isLoading && (!data || data.timestamps?.length === 0)) {
    return (
      <div className="eeg-chart-container">
        <div className="loading">
          <div className="loading-spinner"></div>
          <h3>Loading EEG Data...</h3>
          <p>Connecting to Elasticsearch and fetching data for experiment: {experimentId}</p>
        </div>
      </div>
    );
  }

  if (!data || data.timestamps?.length === 0) {
    return (
      <div className="eeg-chart-container">
        <div className="no-data">
          <h3>No Data Available</h3>
          <p>No EEG data found for experiment: {experimentId}</p>
          <p>Make sure the experiment is running and data is being indexed to Elasticsearch.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="eeg-chart-container">
      <div className="chart-info">
        <div className="data-stats">
          <span>Data Points: {data.timestamps.length}</span>
          <span>Channels: {data.channels?.length || 0}</span>
          {data.timestamps.length > 0 && (
            <span>Latest: {new Date(data.timestamps[data.timestamps.length - 1]).toLocaleTimeString()}</span>
          )}
        </div>
        {isLoading && (
          <div className="updating-indicator">
            <div className="update-dot"></div>
            <span>Updating...</span>
          </div>
        )}
      </div>
      
      <div className="chart-wrapper">
        <Line ref={chartRef} data={chartData} options={options} />
      </div>
    </div>
  );
};

export default EEGChart;
