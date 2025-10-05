import React from 'react';
import Plot from 'react-plotly.js';

interface PieChartProps {
  columnName: string;
  data: { [key: string]: number };
  title?: string;
  maxCategories?: number;
}

const PieChart: React.FC<PieChartProps> = ({ 
  columnName, 
  data, 
  title = "Distribution", 
  maxCategories = 10 
}) => {
  if (!data || Object.keys(data).length === 0) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
        <h6 className="font-medium text-gray-800 mb-2">{title}</h6>
        <div className="text-gray-500 text-center py-4">No data available</div>
      </div>
    );
  }

  // Sort by value and take top categories if specified
  const sortedEntries = Object.entries(data)
    .sort(([, a], [, b]) => b - a)
    .slice(0, maxCategories);

  const labels = sortedEntries.map(([key]) => key);
  const values = sortedEntries.map(([, value]) => value);

  // Calculate percentages manually for better control
  const total = values.reduce((sum, v) => sum + v, 0);
  const formattedPercents = values.map(v => ((v / total) * 100).toFixed(1) + '%');

  const pieTrace = {
    labels,
    values,
    type: 'pie' as const,
    hole: 0.3,
    text: formattedPercents,
    textinfo: 'label+text',
    texttemplate: '%{label}<br>(%{text})',
    textposition: 'outside' as const,
    automargin: true,
    hovertemplate: '%{label}: %{percent} (%{value})<extra></extra>',
    marker: {
      colors: [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
        '#F1948A', '#5DADE2', '#A3E4D7', '#F8C471', '#C39BD3'
      ],
      line: { color: '#fff', width: 2 }
    },
    textfont: {
      size: 12,
      color: '#000'
    }
  };

  const layout = {
    title: `${title} - ${columnName}`,
    showlegend: true,
    height: 450,
    margin: { t: 60, b: 40, l: 40, r: 40 },
    legend: {
      orientation: 'v' as const,
      x: 1.05,
      y: 0.5
    },
    font: { size: 12 },
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
      <Plot
        data={[pieTrace]}
        layout={layout}
        useResizeHandler={true}
        style={{ width: '100%', height: '450px' }}
        config={{ displayModeBar: true, responsive: true }}
      />
    </div>
  );
};

export default PieChart;
