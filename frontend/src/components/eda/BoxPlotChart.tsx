import React from 'react';
import Plot from 'react-plotly.js';

interface BoxPlotChartProps {
  columnName: string;
  data: number[];
  title?: string;
}

const BoxPlotChart: React.FC<BoxPlotChartProps> = ({ 
  columnName, 
  data, 
  title = "Box Plot" 
}) => {
  if (!data || data.length === 0) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
        <h6 className="font-medium text-gray-800 mb-2">{title}</h6>
        <div className="text-gray-500 text-center py-4">No data available</div>
      </div>
    );
  }

  const boxTrace = {
    y: data,
    type: 'box' as const,
    name: columnName,
    boxpoints: 'outliers' as const,
    marker: { color: 'lightgreen' },
    line: { color: 'darkgreen' }
  };

  const layout = {
    title: `${title} - ${columnName}`,
    yaxis: { title: columnName },
    showlegend: false,
    height: 400,
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
      <Plot
        data={[boxTrace]}
        layout={layout}
        useResizeHandler={true}
        style={{ width: '100%', height: '400px' }}
        config={{ displayModeBar: true, responsive: true }}
      />
    </div>
  );
};

export default BoxPlotChart;
