import React from 'react';
import Plot from 'react-plotly.js';

interface CategoricalVariableChartProps {
  columnName: string;
  data: any;
}

const CategoricalVariableChart: React.FC<CategoricalVariableChartProps> = ({ columnName, data }) => {
  if (!data || !data.figure) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
        <h6 className="font-semibold text-lg mb-4 text-gray-800 border-b pb-2">{columnName}</h6>
        <div className="text-gray-500 text-center py-8">
          No data available for {columnName}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
      <h6 className="font-semibold text-lg mb-4 text-gray-800 border-b pb-2">
        📊 {columnName} - Categorical Analysis
      </h6>
      
      {/* Statistics Summary */}
      {data.statistics && (
        <div className="mb-4 bg-green-50 p-4 rounded-lg">
          <h7 className="font-medium text-green-800 mb-2 block">Quick Statistics</h7>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-sm">
            <div><span className="font-medium">Unique Values:</span> {data.statistics.Value[1]}</div>
            <div><span className="font-medium">Most Frequent:</span> {data.statistics.Value[2]}</div>
            <div><span className="font-medium">Missing:</span> {data.statistics.Value[4]}</div>
          </div>
        </div>
      )}
      
      {/* Main Chart */}
      <div className="mb-4">
        <Plot
          data={data.figure.data}
          layout={{
            ...data.figure.layout,
            autosize: true,
            responsive: true,
            height: 800,
          }}
          useResizeHandler={true}
          style={{ width: '100%', height: '800px' }}
          config={{ displayModeBar: true, responsive: true }}
        />
      </div>
    </div>
  );
};

export default CategoricalVariableChart;
