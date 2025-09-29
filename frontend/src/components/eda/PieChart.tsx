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

  // Sort by value and take top categories
  const sortedEntries = Object.entries(data)
    .sort(([,a], [,b]) => b - a)
    .slice(0, maxCategories);

  // Group remaining categories as "Others" if there are more
  const remainingEntries = Object.entries(data).slice(maxCategories);
  if (remainingEntries.length > 0) {
    const othersSum = remainingEntries.reduce((sum, [, value]) => sum + value, 0);
    sortedEntries.push(['Others', othersSum]);
  }

  const labels = sortedEntries.map(([key]) => key);
  const values = sortedEntries.map(([, value]) => value);

  const pieTrace = {
    labels,
    values,
    type: 'pie' as const,
    hole: 0.3,
    textinfo: 'label+percent',
    textposition: 'outside',
    marker: {
      colors: [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
      ]
    }
  };

  const layout = {
    title: `${title} - ${columnName}`,
    showlegend: true,
    height: 400,
    legend: {
      orientation: 'v' as const,
      x: 1.02,
      y: 0.5
    }
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
      <Plot
        data={[pieTrace]}
        layout={layout}
        useResizeHandler={true}
        style={{ width: '100%', height: '400px' }}
        config={{ displayModeBar: true, responsive: true }}
      />
    </div>
  );
};

export default PieChart;
