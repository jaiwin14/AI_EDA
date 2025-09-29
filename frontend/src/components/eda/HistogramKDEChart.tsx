import React from 'react';
import Plot from 'react-plotly.js';

interface HistogramKDEChartProps {
  columnName: string;
  data: number[];
  title?: string;
}

const HistogramKDEChart: React.FC<HistogramKDEChartProps> = ({ 
  columnName, 
  data, 
  title = "Histogram with KDE" 
}) => {
  if (!data || data.length === 0) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
        <h6 className="font-medium text-gray-800 mb-2">{title}</h6>
        <div className="text-gray-500 text-center py-4">No data available</div>
      </div>
    );
  }

  // Create histogram trace
  const histogramTrace = {
    x: data,
    type: 'histogram' as const,
    name: 'Histogram',
    opacity: 0.7,
    marker: { color: 'lightblue' },
    nbinsx: 30,
  };

  // Calculate basic statistics
  const mean = data.reduce((a, b) => a + b, 0) / data.length;
  const median = [...data].sort((a, b) => a - b)[Math.floor(data.length / 2)];

  const layout = {
    title: `${title} - ${columnName}`,
    xaxis: { title: columnName },
    yaxis: { title: 'Frequency' },
    showlegend: true,
    height: 400,
    annotations: [
      {
        x: mean,
        y: 0,
        xref: 'x',
        yref: 'paper',
        text: `Mean: ${mean.toFixed(2)}`,
        showarrow: true,
        arrowhead: 2,
        arrowcolor: 'red',
        ax: 0,
        ay: -30
      },
      {
        x: median,
        y: 0,
        xref: 'x',
        yref: 'paper',
        text: `Median: ${median.toFixed(2)}`,
        showarrow: true,
        arrowhead: 2,
        arrowcolor: 'green',
        ax: 0,
        ay: -50
      }
    ]
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
      <Plot
        data={[histogramTrace]}
        layout={layout}
        useResizeHandler={true}
        style={{ width: '100%', height: '400px' }}
        config={{ displayModeBar: true, responsive: true }}
      />
    </div>
  );
};

export default HistogramKDEChart;
