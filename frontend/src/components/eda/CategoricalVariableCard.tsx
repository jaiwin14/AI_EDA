import { useState } from 'react';

interface CategoricalVariableCardProps {
  columnName: string;
  data: any;
}

const CategoricalVariableCard: React.FC<CategoricalVariableCardProps> = ({ columnName, data }) => {
  const [showDetails, setShowDetails] = useState(false);
  const [activeTab, setActiveTab] = useState('length');

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
      <h5 className="font-semibold text-lg mb-3 text-purple-700">{columnName}</h5>
      
      {/* Basic Information Grid */}
      <div className="grid grid-cols-4 gap-4 mb-4 text-sm">
        <div className="bg-gray-50 p-2 rounded">
          <span className="font-medium text-gray-600">Distinct:</span>
          <div className="text-gray-800">{data?.basic_info?.distinct || 0}</div>
        </div>
        <div className="bg-gray-50 p-2 rounded">
          <span className="font-medium text-gray-600">Distinct (%):</span>
          <div className="text-gray-800">{data?.basic_info?.distinct_percentage?.toFixed(2) || 0}%</div>
        </div>
        <div className="bg-gray-50 p-2 rounded">
          <span className="font-medium text-gray-600">Unique Ratio:</span>
          <div className="text-gray-800">{data?.basic_info?.unique_ratio ? (data.basic_info.unique_ratio * 100).toFixed(1) + '%' : 'N/A'}</div>
        </div>
        <div className="bg-gray-50 p-2 rounded">
          <span className="font-medium text-gray-600">Missing:</span>
          <div className="text-gray-800">{data?.basic_info?.missing || 0}</div>
        </div>
        <div className="bg-gray-50 p-2 rounded">
          <span className="font-medium text-gray-600">Missing (%):</span>
          <div className="text-gray-800">{data?.basic_info?.missing_percentage?.toFixed(2) || 0}%</div>
        </div>
        <div className="bg-gray-50 p-2 rounded">
          <span className="font-medium text-gray-600">Memory Size:</span>
          <div className="text-gray-800">{data?.basic_info?.memory_size?.toFixed(2) || 0} KB</div>
        </div>
        <div className="bg-gray-50 p-2 rounded">
          <span className="font-medium text-gray-600">Most Frequent:</span>
          <div className="text-gray-800 truncate">{data?.basic_info?.most_frequent || 'N/A'}</div>
        </div>
        <div className="bg-gray-50 p-2 rounded">
          <span className="font-medium text-gray-600">Frequency:</span>
          <div className="text-gray-800">{data?.basic_info?.frequency || 0}</div>
        </div>
      </div>

      {/* Show More Details Button */}
      <button
        onClick={() => setShowDetails(!showDetails)}
        className="bg-purple-500 text-white px-4 py-2 rounded hover:bg-purple-600 transition-colors mb-4"
      >
        {showDetails ? 'Hide Details' : 'Show More Details'}
      </button>

      {/* Expandable Details Section */}
      {showDetails && (
        <div className="border-t pt-4">
          {/* Tab Navigation */}
          <div className="flex space-x-2 mb-4">
            {['length', 'categories'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                  activeTab === tab
                    ? 'bg-purple-500 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          {activeTab === 'length' && (
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="bg-purple-50 p-3 rounded">
                <span className="font-medium text-purple-700">Max Length:</span>
                <div className="text-purple-900">{data?.length_stats?.max_length || 'N/A'}</div>
              </div>
              <div className="bg-purple-50 p-3 rounded">
                <span className="font-medium text-purple-700">Median Length:</span>
                <div className="text-purple-900">{data?.length_stats?.median_length?.toFixed(2) || 'N/A'}</div>
              </div>
              <div className="bg-purple-50 p-3 rounded">
                <span className="font-medium text-purple-700">Mean Length:</span>
                <div className="text-purple-900">{data?.length_stats?.mean_length?.toFixed(2) || 'N/A'}</div>
              </div>
              <div className="bg-purple-50 p-3 rounded">
                <span className="font-medium text-purple-700">Min Length:</span>
                <div className="text-purple-900">{data?.length_stats?.min_length || 'N/A'}</div>
              </div>
            </div>
          )}

          {activeTab === 'categories' && (
            <div className="bg-orange-50 p-4 rounded">
              <h6 className="font-medium text-orange-700 mb-3">Categories (Frequency Table)</h6>
              {data?.categories && data.categories.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm border-collapse border border-gray-300">
                    <thead>
                      <tr className="bg-orange-100">
                        <th className="border border-gray-300 px-3 py-2 text-left font-semibold">Category</th>
                        <th className="border border-gray-300 px-3 py-2 text-left font-semibold">Count</th>
                        <th className="border border-gray-300 px-3 py-2 text-left font-semibold">Frequency (%)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.categories.map((item: any, index: number) => (
                        <tr key={index} className="hover:bg-orange-25">
                          <td className="border border-gray-300 px-3 py-2">{item.category}</td>
                          <td className="border border-gray-300 px-3 py-2">{item.count}</td>
                          <td className="border border-gray-300 px-3 py-2">{item.frequency_percent}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-gray-500">No categories data available</div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default CategoricalVariableCard;
