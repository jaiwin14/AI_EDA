import React from 'react';
import Plot from 'react-plotly.js';

interface BivariateChartProps {
  title: string;
  data: any;
  analysisType: 'numeric_vs_numeric' | 'numeric_vs_categorical' | 'categorical_vs_categorical';
}

const BivariateChart: React.FC<BivariateChartProps> = ({ title, data, analysisType }) => {
  if (!data || !data.figure) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
        <h6 className="font-semibold text-lg mb-4 text-gray-800 border-b pb-2">{title}</h6>
        <div className="text-gray-500 text-center py-8">
          No data available for this analysis
        </div>
      </div>
    );
  }

  const getIcon = () => {
    switch (analysisType) {
      case 'numeric_vs_numeric':
        return '📈';
      case 'numeric_vs_categorical':
        return '📊';
      case 'categorical_vs_categorical':
        return '📋';
      default:
        return '🔗';
    }
  };

  const getDescription = () => {
    switch (analysisType) {
      case 'numeric_vs_numeric':
        return 'Correlation and scatter plot analysis';
      case 'numeric_vs_categorical':
        return 'Box plots, violin plots, and mean comparisons';
      case 'categorical_vs_categorical':
        return 'Cross-tabulation and grouped analysis';
      default:
        return 'Bivariate relationship analysis';
    }
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
      <h6 className="font-semibold text-lg mb-2 text-gray-800 border-b pb-2">
        {getIcon()} {title}
      </h6>
      <p className="text-sm text-gray-600 mb-4">{getDescription()}</p>
      
      {/* Additional Info */}
      {data.numeric_column && data.categorical_column && (
        <div className="mb-4 bg-purple-50 p-3 rounded-lg">
          <div className="text-sm">
            <span className="font-medium">Numeric Variable:</span> {data.numeric_column} | 
            <span className="font-medium ml-2">Categorical Variable:</span> {data.categorical_column}
          </div>
        </div>
      )}
      
      {data.categorical_column_1 && data.categorical_column_2 && (
        <div className="mb-4 bg-green-50 p-3 rounded-lg">
          <div className="text-sm">
            <span className="font-medium">Variable 1:</span> {data.categorical_column_1} | 
            <span className="font-medium ml-2">Variable 2:</span> {data.categorical_column_2}
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
            height: analysisType === 'numeric_vs_numeric' && title.includes('pairplot') ? 800 : 600,
          }}
          useResizeHandler={true}
          style={{ 
            width: '100%', 
            height: analysisType === 'numeric_vs_numeric' && title.includes('pairplot') ? '800px' : '600px' 
          }}
          config={{ displayModeBar: true, responsive: true }}
        />
      </div>
    </div>
  );
};

export default BivariateChart;
