import { useState } from 'react';

interface NumericalVariableCardProps {
  columnName: string;
  data: any;
}

const NumericalVariableCard: React.FC<NumericalVariableCardProps> = ({ columnName, data }) => {
  const [showDetails, setShowDetails] = useState(false);
  const [activeTab, setActiveTab] = useState('quantile');

  if (!data) return null;

  return (
    <div className="border border-gray-300 rounded-lg p-4 mb-4 bg-white shadow-sm">
      {/* Basic Information */}
      <div className="mb-4">
        <h4 className="text-lg font-semibold text-gray-800 mb-3">{data.variable_name || columnName}</h4>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
          <div>
            <span className="font-medium text-gray-600">Data Type:</span>
            <div className="text-blue-600 font-medium">{data.data_type || 'N/A'}</div>
          </div>
          <div>
            <span className="font-medium text-gray-600">Uniform:</span>
            <div className={`font-medium ${data.is_uniform ? 'text-green-600' : 'text-red-600'}`}>
              {data.is_uniform ? 'Yes' : 'No'}
            </div>
          </div>
          <div>
            <span className="font-medium text-gray-600">Unique:</span>
            <div className={`font-medium ${data.is_unique ? 'text-green-600' : 'text-red-600'}`}>
              {data.is_unique ? 'Yes' : 'No'}
            </div>
          </div>
          <div>
            <span className="font-medium text-gray-600">Distinct Count:</span>
            <div className="text-purple-600 font-medium">{data.distinct_count?.toLocaleString() || 'N/A'}</div>
          </div>
          <div>
            <span className="font-medium text-gray-600">Distinct %:</span>
            <div className="text-purple-600 font-medium">{data.distinct_percentage || 0}%</div>
          </div>
          <div>
            <span className="font-medium text-gray-600">Missing Count:</span>
            <div className="text-red-600 font-medium">{data.missing_count?.toLocaleString() || 0}</div>
          </div>
          <div>
            <span className="font-medium text-gray-600">Missing %:</span>
            <div className="text-red-600 font-medium">{data.missing_percentage || 0}%</div>
          </div>
          <div>
            <span className="font-medium text-gray-600">Infinite Count:</span>
            <div className="text-orange-600 font-medium">{data.infinite_count?.toLocaleString() || 0}</div>
          </div>
          <div>
            <span className="font-medium text-gray-600">Infinite %:</span>
            <div className="text-orange-600 font-medium">{data.infinite_percentage || 0}%</div>
          </div>
          <div>
            <span className="font-medium text-gray-600">Minimum:</span>
            <div className="text-indigo-600 font-medium">{data.minimum !== null ? Number(data.minimum).toFixed(3) : 'N/A'}</div>
          </div>
          <div>
            <span className="font-medium text-gray-600">Maximum:</span>
            <div className="text-indigo-600 font-medium">{data.maximum !== null ? Number(data.maximum).toFixed(3) : 'N/A'}</div>
          </div>
          <div>
            <span className="font-medium text-gray-600">Zeros:</span>
            <div className="text-gray-600 font-medium">{data.zeros_count?.toLocaleString() || 0}</div>
          </div>
          <div>
            <span className="font-medium text-gray-600">Zeros %:</span>
            <div className="text-gray-600 font-medium">{data.zeros_percentage || 0}%</div>
          </div>
          <div>
            <span className="font-medium text-gray-600">Negative:</span>
            <div className="text-red-600 font-medium">{data.negative_count?.toLocaleString() || 0}</div>
          </div>
          <div>
            <span className="font-medium text-gray-600">Negative %:</span>
            <div className="text-red-600 font-medium">{data.negative_percentage || 0}%</div>
          </div>
          <div>
            <span className="font-medium text-gray-600">Memory Size:</span>
            <div className="text-teal-600 font-medium">
              {data.memory_size_bytes ? `${(data.memory_size_bytes / 1024).toFixed(2)} KB` : 'N/A'}
            </div>
          </div>
        </div>
      </div>

      {/* Details Toggle Button */}
      <button
        onClick={() => setShowDetails(!showDetails)}
        className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded text-sm font-medium transition-colors"
      >
        {showDetails ? 'Hide Details' : 'Show More Details'}
      </button>

      {/* Expandable Details Section */}
      {showDetails && (
        <div className="mt-4 border-t border-gray-200 pt-4">
          {/* Tab Navigation */}
          <div className="flex flex-wrap gap-2 mb-4">
            {['quantile', 'descriptive', 'common', 'extreme'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                  activeTab === tab
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {tab === 'quantile' && 'Quantile Statistics'}
                {tab === 'descriptive' && 'Descriptive Statistics'}
                {tab === 'common' && 'Common Values'}
                {tab === 'extreme' && 'Extreme Values'}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div className="bg-gray-50 p-4 rounded">
            {activeTab === 'quantile' && data.quantile_statistics && (
              <div>
                <h5 className="font-semibold mb-3">Quantile Statistics</h5>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
                  <div>
                    <span className="font-medium text-gray-600">Minimum:</span>
                    <div className="font-medium">{data.quantile_statistics.minimum?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="font-medium text-gray-600">5th Percentile:</span>
                    <div className="font-medium">{data.quantile_statistics.percentile_5?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="font-medium text-gray-600">Q1:</span>
                    <div className="font-medium">{data.quantile_statistics.q1?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="font-medium text-gray-600">Median:</span>
                    <div className="font-medium">{data.quantile_statistics.median?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="font-medium text-gray-600">Q3:</span>
                    <div className="font-medium">{data.quantile_statistics.q3?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="font-medium text-gray-600">95th Percentile:</span>
                    <div className="font-medium">{data.quantile_statistics.percentile_95?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="font-medium text-gray-600">Maximum:</span>
                    <div className="font-medium">{data.quantile_statistics.maximum?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="font-medium text-gray-600">Range:</span>
                    <div className="font-medium">{data.quantile_statistics.range?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="font-medium text-gray-600">IQR:</span>
                    <div className="font-medium">{data.quantile_statistics.iqr?.toFixed(3) || 'N/A'}</div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'descriptive' && data.descriptive_statistics && (
              <div>
                <h5 className="font-semibold mb-3">Descriptive Statistics</h5>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
                  <div>
                    <span className="font-medium text-gray-600">Standard Deviation:</span>
                    <div className="font-medium">{data.descriptive_statistics.standard_deviation?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="font-medium text-gray-600">CV:</span>
                    <div className="font-medium">{data.descriptive_statistics.coefficient_of_variation || 0}%</div>
                  </div>
                  <div>
                    <span className="font-medium text-gray-600">Kurtosis:</span>
                    <div className="font-medium">{data.descriptive_statistics.kurtosis?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="font-medium text-gray-600">Mean:</span>
                    <div className="font-medium">{data.descriptive_statistics.mean?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="font-medium text-gray-600">MAD:</span>
                    <div className="font-medium">{data.descriptive_statistics.median_absolute_deviation?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="font-medium text-gray-600">Skewness:</span>
                    <div className="font-medium">{data.descriptive_statistics.skewness?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="font-medium text-gray-600">Sum:</span>
                    <div className="font-medium">{data.descriptive_statistics.sum?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="font-medium text-gray-600">Variance:</span>
                    <div className="font-medium">{data.descriptive_statistics.variance?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="font-medium text-gray-600">Monotonicity:</span>
                    <div className="font-medium">{data.descriptive_statistics.monotonicity || 'N/A'}</div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'common' && data.common_values && (
              <div>
                <h5 className="font-semibold mb-3">Most Frequently Occurring Values</h5>
                <div className="space-y-2">
                  {data.common_values.values?.map((item: any, index: number) => (
                    <div key={index} className="flex justify-between items-center bg-white p-2 rounded border">
                      <span className="font-medium">{item.value?.toFixed(3)}</span>
                      <span className="text-sm text-gray-600">
                        Count: {item.count?.toLocaleString()} ({item.percentage}%)
                      </span>
                    </div>
                  )) || <div className="text-gray-500">No common values data available</div>}
                </div>
              </div>
            )}

            {activeTab === 'extreme' && data.extreme_values && (
              <div>
                <h5 className="font-semibold mb-3">Extreme Values</h5>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h6 className="font-medium mb-2 text-red-600">Lowest Values</h6>
                    <div className="space-y-1">
                      {data.extreme_values.lowest?.map((item: any, index: number) => (
                        <div key={index} className="flex justify-between items-center bg-red-50 p-2 rounded text-sm">
                          <span className="font-medium">{item.value?.toFixed(3)}</span>
                          <span className="text-gray-600">Index: {item.index}</span>
                        </div>
                      )) || <div className="text-gray-500">No data available</div>}
                    </div>
                  </div>
                  <div>
                    <h6 className="font-medium mb-2 text-green-600">Highest Values</h6>
                    <div className="space-y-1">
                      {data.extreme_values.highest?.map((item: any, index: number) => (
                        <div key={index} className="flex justify-between items-center bg-green-50 p-2 rounded text-sm">
                          <span className="font-medium">{item.value?.toFixed(3)}</span>
                          <span className="text-gray-600">Index: {item.index}</span>
                        </div>
                      )) || <div className="text-gray-500">No data available</div>}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default NumericalVariableCard;
