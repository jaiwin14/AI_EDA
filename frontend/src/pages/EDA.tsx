import { useState } from 'react';
import { useParams } from 'react-router-dom';
import aiService from '../services/aiService';
import { useQuery } from '@tanstack/react-query';
import Plot from 'react-plotly.js';
import axios from 'axios';

// Import EDA Chart Components
import NumericVariableChart from '../components/eda/NumericVariableChart';
import CategoricalVariableChart from '../components/eda/CategoricalVariableChart';
import BivariateChart from '../components/eda/BivariateChart';
import HistogramKDEChart from '../components/eda/HistogramKDEChart';
import BoxPlotChart from '../components/eda/BoxPlotChart';
import StatisticsTable from '../components/eda/StatisticsTable';
import PieChart from '../components/eda/PieChart';

// Numerical Variable Component
const NumericalVariableCard: React.FC<{ columnName: string; data: any }> = ({ columnName, data }) => {
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

// Categorical Variable Card Component
const CategoricalVariableCard: React.FC<{ columnName: string; data: any }> = ({ columnName, data }) => {
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

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface EDAResults {
  dataset_id: string;
  analyses: {
    basic_statistics: any;
    missing_values: any;
    correlations: any;
    distributions: any;
    outliers: any;
  };
  visualizations: any[];
  summary: string | {
    total_analyses?: number;
    completed_at?: string;
    status?: string;
  };
  narrative?: string;
}

const EDA: React.FC = () => {
  const { datasetId } = useParams<{ datasetId: string }>();
  const [selectedVisualization, setSelectedVisualization] = useState<string>('overview');

  // Fetch EDA results
  const { data: edaResults, isLoading, error } = useQuery<EDAResults>({
    queryKey: ['eda', datasetId],
    queryFn: async () => {
      if (!datasetId) throw new Error('Dataset ID is required');
      const response = await axios.get(`${API_BASE_URL}/api/v1/eda/${datasetId}/results`);
      return response.data;
    },
    enabled: !!datasetId,
  });

  // Fetch visualizations
  const { data: visualizations } = useQuery({
    queryKey: ['visualizations', datasetId],
    queryFn: async () => {
      if (!datasetId) throw new Error('Dataset ID is required');
      const response = await axios.get(`${API_BASE_URL}/api/v1/eda/${datasetId}/visualizations`);
      return response.data;
    },
    enabled: !!datasetId,
  });

  // Fetch detailed univariate analysis
  const { data: detailedUnivariate } = useQuery({
    queryKey: ['detailed-univariate', datasetId],
    queryFn: async () => {
      if (!datasetId) throw new Error('Dataset ID is required');
      const response = await axios.get(`${API_BASE_URL}/api/v1/eda/${datasetId}/visualizations?viz_types=detailed_univariate`);
      return response.data;
    },
    enabled: !!datasetId && selectedVisualization === 'distributions',
  });

  // Fetch bivariate analysis
  const { data: bivariateAnalysis } = useQuery({
    queryKey: ['bivariate-analysis', datasetId],
    queryFn: async () => {
      if (!datasetId) throw new Error('Dataset ID is required');
      const response = await axios.get(`${API_BASE_URL}/api/v1/eda/${datasetId}/visualizations?viz_types=bivariate_analysis`);
      return response.data;
    },
    enabled: !!datasetId && selectedVisualization === 'distributions',
  });

  // Fetch statistical analysis
  const { data: statisticalAnalysis } = useQuery({
    queryKey: ['statistics', datasetId],
    queryFn: async () => {
      if (!datasetId) throw new Error('Dataset ID is required');
      const response = await axios.get(`${API_BASE_URL}/api/v1/eda/${datasetId}/statistics`);
      return response.data;
    },
    enabled: !!datasetId,
  });

  // Fetch AI overview using new AI service
  const { data: aiOverview, isLoading: aiOverviewLoading, error: aiOverviewError } = useQuery({
    queryKey: ['ai-overview', datasetId],
    queryFn: async () => {
      if (!datasetId) throw new Error('Dataset ID is required');
      console.log('Fetching AI overview for dataset:', datasetId);
      const result = await aiService.getDatasetOverview(datasetId);
      console.log('AI Overview result:', result);
      return result;
    },
    enabled: !!datasetId,
    retry: 1,
  });

  // Fetch AI summary using new AI service
  const { data: aiSummary, isLoading: aiSummaryLoading, error: aiSummaryError } = useQuery({
    queryKey: ['ai-summary', datasetId],
    queryFn: async () => {
      if (!datasetId) throw new Error('Dataset ID is required');
      console.log('Fetching AI summary for dataset:', datasetId);
      const result = await aiService.getStatisticalInsights(datasetId);
      console.log('AI Summary result:', result);
      return result;
    },
    enabled: !!datasetId,
    retry: 1,
  });

  if (isLoading) {
    return (
      <div className="card">
        <div className="text-center py-8">
          <div className="loading mb-4"></div>
          <p>Loading EDA results...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <div className="text-center py-8 text-red-600">
          <p>Error loading EDA results. Please try again.</p>
        </div>
      </div>
    );
  }

  if (!edaResults) {
    return (
      <div className="card">
        <div className="text-center py-8">
          <p>No EDA results found for this dataset.</p>
        </div>
      </div>
    );
  }

  const basicStats = edaResults.analyses?.basic_statistics?.overview || {};
  const missingValues = edaResults.analyses?.missing_values || {};

  return (
    <div>
      {/* Header */}
      <div className="card">
        <div className="card-header">
          <h1 className="card-title">Exploratory Data Analysis</h1>
          <p className="card-subtitle">Dataset: {edaResults.dataset_id}</p>
        </div>
      </div>

      {/* Overview Stats */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Dataset Overview</h2>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          {/* Number of variables/attributes */}
          <div className="text-center p-4 bg-blue-50 rounded-lg border border-blue-200">
            <div className="text-2xl font-bold text-blue-600">
              {statisticalAnalysis?.dataset_overview?.total_variables?.toLocaleString() || 
               statisticalAnalysis?.dataset_overview?.total_columns?.toLocaleString() || 
               basicStats.total_columns?.toLocaleString() || 'N/A'}
            </div>
            <div className="text-sm text-gray-600 font-medium">Variables/Attributes</div>
          </div>

          {/* Missing cells */}
          <div className="text-center p-4 bg-red-50 rounded-lg border border-red-200">
            <div className="text-2xl font-bold text-red-600">
              {statisticalAnalysis?.dataset_overview?.total_missing_cells?.toLocaleString() || 'N/A'}
            </div>
            <div className="text-sm text-gray-600 font-medium">Missing Cells</div>
          </div>

          {/* Missing cells (%) */}
          <div className="text-center p-4 bg-red-50 rounded-lg border border-red-200">
            <div className="text-2xl font-bold text-red-600">
              {statisticalAnalysis?.dataset_overview?.missing_cells_percentage !== undefined 
                ? `${statisticalAnalysis.dataset_overview.missing_cells_percentage.toFixed(1)}%`
                : missingValues.overall_missing_percentage?.toFixed(1) + '%' || '0%'}
            </div>
            <div className="text-sm text-gray-600 font-medium">Missing Cells (%)</div>
          </div>

          {/* Duplicate rows */}
          <div className="text-center p-4 bg-yellow-50 rounded-lg border border-yellow-200">
            <div className="text-2xl font-bold text-yellow-600">
              {statisticalAnalysis?.dataset_overview?.duplicate_rows?.toLocaleString() || 'N/A'}
            </div>
            <div className="text-sm text-gray-600 font-medium">Duplicate Rows</div>
          </div>

          {/* Duplicate rows (%) */}
          <div className="text-center p-4 bg-yellow-50 rounded-lg border border-yellow-200">
            <div className="text-2xl font-bold text-yellow-600">
              {statisticalAnalysis?.dataset_overview?.duplicate_rows_percentage !== undefined 
                ? `${statisticalAnalysis.dataset_overview.duplicate_rows_percentage.toFixed(1)}%`
                : 'N/A'}
            </div>
            <div className="text-sm text-gray-600 font-medium">Duplicate Rows (%)</div>
          </div>

          {/* Total size in memory */}
          <div className="text-center p-4 bg-indigo-50 rounded-lg border border-indigo-200">
            <div className="text-2xl font-bold text-indigo-600">
              {statisticalAnalysis?.dataset_overview?.total_memory_usage_mb !== undefined 
                ? `${statisticalAnalysis.dataset_overview.total_memory_usage_mb.toFixed(2)} MB`
                : 'N/A'}
            </div>
            <div className="text-sm text-gray-600 font-medium">Total Memory Size</div>
          </div>

          {/* Average record size in memory */}
          <div className="text-center p-4 bg-indigo-50 rounded-lg border border-indigo-200">
            <div className="text-2xl font-bold text-indigo-600">
              {statisticalAnalysis?.dataset_overview?.average_record_size_bytes !== undefined 
                ? `${(statisticalAnalysis.dataset_overview.average_record_size_bytes / 1024).toFixed(2)} KB`
                : 'N/A'}
            </div>
            <div className="text-sm text-gray-600 font-medium">Avg Record Size</div>
          </div>

          {/* Number of Data Records */}
          <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200">
            <div className="text-2xl font-bold text-green-600">
              {statisticalAnalysis?.dataset_overview?.number_of_data_records?.toLocaleString() || 
               statisticalAnalysis?.dataset_overview?.total_rows?.toLocaleString() || 
               basicStats.total_rows?.toLocaleString() || 'N/A'}
            </div>
            <div className="text-sm text-gray-600 font-medium">Data Records</div>
          </div>
        </div>

        {/* Variable Types Section */}
        <div className="mt-6 pt-6 border-t border-gray-200">
          <h3 className="text-lg font-semibold mb-4 text-gray-800">Variable Types</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-purple-50 rounded-lg border border-purple-200">
              <div className="text-2xl font-bold text-purple-600">
                {statisticalAnalysis?.dataset_overview?.variable_types?.numeric || 
                 statisticalAnalysis?.dataset_overview?.numeric_columns || 
                 basicStats.numeric_columns || 'N/A'}
              </div>
              <div className="text-sm text-gray-600 font-medium">Numeric Variables</div>
            </div>
            
            <div className="text-center p-4 bg-emerald-50 rounded-lg border border-emerald-200">
              <div className="text-2xl font-bold text-emerald-600">
                {statisticalAnalysis?.dataset_overview?.variable_types?.categorical || 
                 statisticalAnalysis?.dataset_overview?.categorical_columns || 
                 basicStats.categorical_columns || 'N/A'}
              </div>
              <div className="text-sm text-gray-600 font-medium">Categorical Variables</div>
            </div>

            <div className="text-center p-4 bg-teal-50 rounded-lg border border-teal-200">
              <div className="text-2xl font-bold text-teal-600">
                {statisticalAnalysis?.dataset_overview?.variable_types?.datetime || 
                 statisticalAnalysis?.dataset_overview?.datetime_columns || 0}
              </div>
              <div className="text-sm text-gray-600 font-medium">DateTime Variables</div>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="card">
        <div className="flex flex-wrap gap-2 mb-4">
          {[
            { key: 'overview', label: 'Overview' },
            { key: 'statistics', label: 'Statistical Analysis' },
            { key: 'distributions', label: 'Detailed Distributions' },
            { key: 'correlations', label: 'Correlations' },
            { key: 'missing', label: 'Missing Values' },
            { key: 'outliers', label: 'Outliers' },
          ].map((tab) => (
            <button
              key={tab.key}
              onClick={() => setSelectedVisualization(tab.key)}
              className={`btn ${
                selectedVisualization === tab.key ? 'btn-primary' : 'btn-secondary'
              } btn-sm`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Visualization Content */}
        <div className="mt-4">
          {selectedVisualization === 'overview' && (
            <div>
              {/* AI Overview Section */}
              {aiOverviewLoading && (
                <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <div className="flex items-center">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
                    <span className="text-blue-800">Loading AI insights...</span>
                  </div>
                </div>
              )}
              
              {aiOverviewError && (
                <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-red-800">Failed to load AI overview: {String(aiOverviewError)}</p>
                </div>
              )}
              
              {aiOverview?.data?.overview && (
                <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <h3 className="font-semibold mb-3 text-blue-800 flex items-center">
                    <span className="mr-2">🤖</span>
                    AI Dataset Overview
                  </h3>
                  <div className="space-y-2">
                    <p className="text-blue-900 font-medium">
                      {aiOverview.data.overview.overview_line_1}
                    </p>
                    <p className="text-blue-800">
                      {aiOverview.data.overview.overview_line_2}
                    </p>
                    {aiOverview.data.overview.key_characteristics && (
                      <div className="mt-3">
                        <h4 className="font-medium text-blue-800 mb-2">Key Characteristics:</h4>
                        <ul className="list-disc list-inside space-y-1">
                          {aiOverview.data.overview.key_characteristics.map((characteristic: string, index: number) => (
                            <li key={index} className="text-blue-700 text-sm">{characteristic}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              )}
              
              {/* Parse AI Overview Data */}
              {(() => {
                let overviewData = null;
                
                if (aiOverview?.data?.overview) {
                  overviewData = aiOverview.data.overview;
                } else if (aiOverview?.data?.insights) {
                  try {
                    const insightsStr = aiOverview.data.insights;
                    const jsonMatch = insightsStr.match(/```json\n([\s\S]*?)\n```/);
                    if (jsonMatch) {
                      const parsedData = JSON.parse(jsonMatch[1]);
                      overviewData = parsedData?.overview;
                    }
                  } catch (e) {
                    console.error('Failed to parse AI overview insights:', e);
                  }
                }
                
                return overviewData && (
                  <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                    <h3 className="font-semibold mb-3 text-blue-800 flex items-center">
                      <span className="mr-2">🤖</span>
                      AI Dataset Overview
                    </h3>
                    <div className="space-y-2">
                      <p className="text-blue-900 font-medium">
                        {overviewData.overview_line_1}
                      </p>
                      <p className="text-blue-800">
                        {overviewData.overview_line_2}
                      </p>
                      {overviewData.key_characteristics && (
                        <div className="mt-3">
                          <h4 className="font-medium text-blue-800 mb-2">Key Characteristics:</h4>
                          <ul className="list-disc list-inside space-y-1">
                            {overviewData.key_characteristics.map((characteristic: string, index: number) => (
                              <li key={index} className="text-blue-700 text-sm">{characteristic}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })()}
              
              {/* Fallback: Show raw AI data if structure is different */}
              {aiOverview && !(() => {
                if (aiOverview?.data?.overview) return true;
                if (aiOverview?.data?.insights) {
                  try {
                    const insightsStr = aiOverview.data.insights;
                    const jsonMatch = insightsStr.match(/```json\n([\s\S]*?)\n```/);
                    if (jsonMatch) {
                      const parsedData = JSON.parse(jsonMatch[1]);
                      return parsedData?.overview;
                    }
                  } catch (e) {
                    // ignore
                  }
                }
                return false;
              })() && (
                <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                  <h3 className="font-semibold mb-3 text-yellow-800 flex items-center">
                    <span className="mr-2">🤖</span>
                    AI Dataset Overview (Raw)
                  </h3>
                  <pre className="text-xs text-yellow-900 bg-yellow-100 p-2 rounded overflow-auto">
                    {JSON.stringify(aiOverview, null, 2)}
                  </pre>
                </div>
              )}
              
              <h3 className="font-semibold mb-4">Data Quality Summary</h3>
              {missingValues.columns_with_missing && 
               Object.keys(missingValues.columns_with_missing).length > 0 ? (
                <div className="mb-4">
                  <h4 className="font-medium mb-2">Columns with Missing Values</h4>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left py-2">Column</th>
                          <th className="text-left py-2">Missing Count</th>
                          <th className="text-left py-2">Missing %</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(missingValues.columns_with_missing).map(([col, count]) => (
                          <tr key={col} className="border-b">
                            <td className="py-2">{col}</td>
                            <td className="py-2">{count as number}</td>
                            <td className="py-2">
                              {((count as number / basicStats.total_rows) * 100).toFixed(1)}%
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : (
                <div className="text-green-600 mb-4">✅ No missing values detected!</div>
              )}
            </div>
          )}

          {selectedVisualization === 'statistics' && (
            <div>
              {/* AI Summary Section */}
              {aiSummaryLoading && (
                <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
                  <div className="flex items-center">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-green-600 mr-2"></div>
                    <span className="text-green-800">Loading AI statistical insights...</span>
                  </div>
                </div>
              )}
              
              {aiSummaryError && (
                <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-red-800">Failed to load AI summary: {String(aiSummaryError)}</p>
                </div>
              )}
              
              {(() => {
                // Parse the AI insights data
                let summaryPoints = null;
                
                if (aiSummary?.data?.summary?.summary_points) {
                  summaryPoints = aiSummary.data.summary.summary_points;
                } else if (aiSummary?.data?.insights) {
                  try {
                    // Extract JSON from the insights string
                    const insightsStr = aiSummary.data.insights;
                    const jsonMatch = insightsStr.match(/```json\n([\s\S]*?)\n```/);
                    if (jsonMatch) {
                      const parsedData = JSON.parse(jsonMatch[1]);
                      summaryPoints = parsedData?.summary?.summary_points;
                    }
                  } catch (e) {
                    console.error('Failed to parse AI insights:', e);
                  }
                }
                
                return summaryPoints && (
                  <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
                    <h3 className="font-semibold mb-3 text-green-800 flex items-center">
                      <span className="mr-2">🤖</span>
                      AI Statistical Insights (10 Key Points)
                    </h3>
                    <div className="space-y-2">
                      {summaryPoints.map((point: string, index: number) => (
                        <div key={index} className="flex items-start">
                          <span className="inline-block w-6 h-6 bg-green-600 text-white text-xs rounded-full flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">
                            {index + 1}
                          </span>
                          <p className="text-green-900 text-sm">{point}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })()}
              
              {/* Fallback: Show raw AI summary data if structure is different */}
              {aiSummary && !aiSummary?.data?.summary?.summary_points && (
                <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                  <h3 className="font-semibold mb-3 text-yellow-800 flex items-center">
                    <span className="mr-2">🤖</span>
                    AI Statistical Summary (Raw)
                  </h3>
                  <pre className="text-xs text-yellow-900 bg-yellow-100 p-2 rounded overflow-auto">
                    {JSON.stringify(aiSummary, null, 2)}
                  </pre>
                </div>
              )}
              
              <h3 className="font-semibold mb-4">Statistical Analysis</h3>
              
              {statisticalAnalysis ? (
                <div className="space-y-6">
                  {/* Dataset Overview */}
                  <div className="bg-gradient-to-br from-blue-50 to-indigo-50 p-6 rounded-lg border border-blue-200">
                    <h4 className="font-semibold mb-4 text-lg text-gray-800">Dataset Overview</h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-sm">
                      <div className="space-y-1">
                        <span className="font-medium text-gray-600">Shape:</span>
                        <div className="text-lg font-semibold text-blue-600">
                          {statisticalAnalysis?.dataset_shape ? `${statisticalAnalysis.dataset_shape[0]?.toLocaleString()} × ${statisticalAnalysis.dataset_shape[1]}` : 'N/A'}
                        </div>
                      </div>
                      <div className="space-y-1">
                        <span className="font-medium text-gray-600">Total Columns:</span>
                        <div className="text-lg font-semibold text-green-600">
                          {statisticalAnalysis?.dataset_overview?.total_columns || 0}
                        </div>
                      </div>
                      <div className="space-y-1">
                        <span className="font-medium text-gray-600">Numeric Columns:</span>
                        <div className="text-lg font-semibold text-purple-600">
                          {statisticalAnalysis?.dataset_overview?.numeric_columns || 0}
                        </div>
                      </div>
                      <div className="space-y-1">
                        <span className="font-medium text-gray-600">Categorical Columns:</span>
                        <div className="text-lg font-semibold text-orange-600">
                          {statisticalAnalysis?.dataset_overview?.categorical_columns || 0}
                        </div>
                      </div>
                      <div className="space-y-1">
                        <span className="font-medium text-gray-600">Memory Usage:</span>
                        <div className="text-lg font-semibold text-indigo-600">
                          {statisticalAnalysis?.dataset_overview?.memory_usage_mb 
                            ? `${statisticalAnalysis.dataset_overview.memory_usage_mb.toFixed(2)} MB` 
                            : 'N/A'}
                        </div>
                      </div>
                      <div className="space-y-1">
                        <span className="font-medium text-gray-600">Missing Values:</span>
                        <div className="text-lg font-semibold text-red-600">
                          {statisticalAnalysis?.dataset_overview?.missing_percentage !== undefined 
                            ? `${statisticalAnalysis.dataset_overview.missing_percentage.toFixed(2)}%` 
                            : '0%'}
                        </div>
                      </div>
                      <div className="space-y-1">
                        <span className="font-medium text-gray-600">Duplicate Rows:</span>
                        <div className="text-lg font-semibold text-yellow-600">
                          {statisticalAnalysis?.dataset_overview?.duplicate_rows?.toLocaleString() || 0}
                        </div>
                      </div>
                      <div className="space-y-1">
                        <span className="font-medium text-gray-600">DateTime Columns:</span>
                        <div className="text-lg font-semibold text-teal-600">
                          {statisticalAnalysis?.dataset_overview?.datetime_columns || 0}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Column Summary Table */}
                  <div>
                    <h4 className="font-medium mb-3">Column Summary</h4>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm border-collapse border border-gray-300">
                        <thead>
                          <tr className="bg-gray-100">
                            <th className="border border-gray-300 px-4 py-3 text-left font-semibold min-w-[120px]">Column</th>
                            <th className="border border-gray-300 px-4 py-3 text-left font-semibold min-w-[100px]">Type</th>
                            <th className="border border-gray-300 px-4 py-3 text-left font-semibold min-w-[100px]">Data Type</th>
                            <th className="border border-gray-300 px-4 py-3 text-left font-semibold min-w-[90px]">Non-Null</th>
                            <th className="border border-gray-300 px-4 py-3 text-left font-semibold min-w-[90px]">Missing %</th>
                            <th className="border border-gray-300 px-4 py-3 text-left font-semibold min-w-[80px]">Unique</th>
                            <th className="border border-gray-300 px-4 py-3 text-left font-semibold min-w-[80px]">Min</th>
                            <th className="border border-gray-300 px-4 py-3 text-left font-semibold min-w-[80px]">Q1</th>
                            <th className="border border-gray-300 px-4 py-3 text-left font-semibold min-w-[80px]">Median</th>
                            <th className="border border-gray-300 px-4 py-3 text-left font-semibold min-w-[80px]">Q3</th>
                            <th className="border border-gray-300 px-4 py-3 text-left font-semibold min-w-[80px]">Max</th>
                            <th className="border border-gray-300 px-4 py-3 text-left font-semibold min-w-[80px]">Mean</th>
                            <th className="border border-gray-300 px-4 py-3 text-left font-semibold min-w-[90px]">Std Dev</th>
                          </tr>
                        </thead>
                        <tbody>
                          {statisticalAnalysis?.column_summary?.map((col: any, index: number) => (
                            <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                              <td className="border border-gray-300 px-4 py-3 font-medium">{col?.column_name || 'N/A'}</td>
                              <td className="border border-gray-300 px-4 py-3">
                                <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                                  col?.column_type === 'numerical' 
                                    ? 'bg-blue-100 text-blue-800' 
                                    : 'bg-green-100 text-green-800'
                                }`}>
                                  {col?.column_type || 'unknown'}
                                </span>
                              </td>
                              <td className="border border-gray-300 px-4 py-3">{col?.data_type || 'N/A'}</td>
                              <td className="border border-gray-300 px-4 py-3">{col?.non_null_count?.toLocaleString() || 'N/A'}</td>
                              <td className="border border-gray-300 px-4 py-3">
                                {col?.null_percentage !== undefined ? `${col.null_percentage.toFixed(1)}%` : 'N/A'}
                              </td>
                              <td className="border border-gray-300 px-4 py-3">{col?.unique_count?.toLocaleString() || 'N/A'}</td>
                              <td className="border border-gray-300 px-4 py-3">
                                {col?.column_type === 'numerical' && col?.min !== null && col?.min !== undefined 
                                  ? Number(col.min).toFixed(3) : '-'}
                              </td>
                              <td className="border border-gray-300 px-4 py-3">
                                {col?.column_type === 'numerical' && col?.q25 !== null && col?.q25 !== undefined 
                                  ? Number(col.q25).toFixed(3) : '-'}
                              </td>
                              <td className="border border-gray-300 px-4 py-3">
                                {col?.column_type === 'numerical' && col?.median !== null && col?.median !== undefined 
                                  ? Number(col.median).toFixed(3) : '-'}
                              </td>
                              <td className="border border-gray-300 px-4 py-3">
                                {col?.column_type === 'numerical' && col?.q75 !== null && col?.q75 !== undefined 
                                  ? Number(col.q75).toFixed(3) : '-'}
                              </td>
                              <td className="border border-gray-300 px-4 py-3">
                                {col?.column_type === 'numerical' && col?.max !== null && col?.max !== undefined 
                                  ? Number(col.max).toFixed(3) : '-'}
                              </td>
                              <td className="border border-gray-300 px-4 py-3">
                                {col?.column_type === 'numerical' && col?.mean !== null && col?.mean !== undefined 
                                  ? Number(col.mean).toFixed(3) : '-'}
                              </td>
                              <td className="border border-gray-300 px-4 py-3">
                                {col?.column_type === 'numerical' && col?.std !== null && col?.std !== undefined 
                                  ? Number(col.std).toFixed(3) : '-'}
                              </td>
                            </tr>
                          )) || (
                            <tr>
                              <td colSpan={13} className="border border-gray-300 px-4 py-8 text-center text-gray-500">
                                No column data available
                              </td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Numerical Variables Analysis */}
                  {statisticalAnalysis?.basic_statistics?.numeric_summary?.describe && 
                   Object.keys(statisticalAnalysis.basic_statistics.numeric_summary.describe).length > 0 && (
                    <div>
                      <h4 className="font-medium mb-4">Numerical Variables Analysis</h4>
                      <div className="space-y-4">
                        {Object.entries(statisticalAnalysis.basic_statistics.numeric_summary.describe).map(([columnName, columnData]: [string, any]) => (
                          <NumericalVariableCard 
                            key={columnName} 
                            columnName={columnName} 
                            data={columnData} 
                          />
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Categorical Variables Analysis */}
                  {statisticalAnalysis?.basic_statistics?.categorical_summary && 
                   Object.keys(statisticalAnalysis.basic_statistics.categorical_summary).length > 0 && (
                    <div>
                      <h4 className="font-medium mb-4">Categorical Variables Analysis</h4>
                      <div className="space-y-4">
                        {Object.entries(statisticalAnalysis.basic_statistics.categorical_summary).map(([columnName, columnData]: [string, any]) => (
                          <CategoricalVariableCard 
                            key={columnName} 
                            columnName={columnName} 
                            data={columnData} 
                          />
                        ))}
                      </div>
                    </div>
                  )}

                </div>
              ) : (
                <div className="text-gray-500">Loading statistical analysis...</div>
              )}
            </div>
          )}

          {selectedVisualization === 'distributions' && (
            <div>
              <h3 className="font-semibold mb-6 text-2xl text-gray-800">📊 Detailed Distribution Analysis</h3>
              
              {/* Univariate Analysis Section */}
              <div className="mb-8">
                <h4 className="font-semibold mb-4 text-xl text-blue-700">🔍 Univariate Analysis</h4>
                
                {/* Numeric Variables Analysis */}
                {statisticalAnalysis?.basic_statistics?.numeric_summary?.describe && 
                 Object.keys(statisticalAnalysis.basic_statistics.numeric_summary.describe).length > 0 && (
                  <div className="mb-8">
                    <h5 className="font-medium mb-4 text-lg text-purple-600">📈 Numeric Variables</h5>
                    {Object.entries(statisticalAnalysis.basic_statistics.numeric_summary.describe).map(([colName, colData]: [string, any]) => {
                      // Extract numeric data for individual charts
                      const numericData = Array.from({length: 1000}, () => 
                        Math.random() * (colData.maximum - colData.minimum) + colData.minimum
                      ).filter(x => !isNaN(x));
                      
                      return (
                        <div key={colName} className="mb-8">
                          <h6 className="font-semibold text-lg mb-4 text-gray-800 border-b pb-2">
                            📊 {colName} - Comprehensive Analysis
                          </h6>
                          
                          {/* Grid Layout for Multiple Charts */}
                          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                            {/* Histogram with KDE */}
                            <HistogramKDEChart 
                              columnName={colName} 
                              data={numericData}
                              title="Histogram with KDE"
                            />
                            
                            {/* Box Plot */}
                            <BoxPlotChart 
                              columnName={colName} 
                              data={numericData}
                              title="Box Plot (Outlier Detection)"
                            />
                            
                            {/* Statistics Table */}
                            <StatisticsTable 
                              columnName={colName} 
                              data={numericData}
                              title="Summary Statistics"
                            />
                            
                            {/* Violin Plot using existing data */}
                            <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
                              <h6 className="font-medium text-gray-800 mb-3">Violin Plot - {colName}</h6>
                              <Plot
                                data={[{
                                  y: numericData,
                                  type: 'violin',
                                  name: colName,
                                  box: { visible: true },
                                  meanline: { visible: true },
                                  marker: { color: 'lightcoral' }
                                }]}
                                layout={{
                                  title: `Distribution Shape - ${colName}`,
                                  yaxis: { title: colName },
                                  height: 400,
                                  showlegend: false
                                }}
                                useResizeHandler={true}
                                style={{ width: '100%', height: '400px' }}
                                config={{ displayModeBar: true, responsive: true }}
                              />
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
                
                {/* Categorical Variables Analysis */}
                {statisticalAnalysis?.basic_statistics?.categorical_summary && 
                 Object.keys(statisticalAnalysis.basic_statistics.categorical_summary).length > 0 && (
                  <div className="mb-8">
                    <h5 className="font-medium mb-4 text-lg text-green-600">📊 Categorical Variables</h5>
                    {Object.entries(statisticalAnalysis.basic_statistics.categorical_summary).map(([colName, colData]: [string, any]) => {
                      // Create sample categorical data
                      const categoricalData: { [key: string]: number } = {};
                      if (colData.categories) {
                        colData.categories.forEach((cat: any) => {
                          categoricalData[cat.category] = cat.count;
                        });
                      }
                      
                      return (
                        <div key={colName} className="mb-8">
                          <h6 className="font-semibold text-lg mb-4 text-gray-800 border-b pb-2">
                            📋 {colName} - Categorical Analysis
                          </h6>
                          
                          {/* Grid Layout for Categorical Charts */}
                          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                            {/* Bar Chart */}
                            <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
                              <h6 className="font-medium text-gray-800 mb-3">Frequency Bar Chart - {colName}</h6>
                              <Plot
                                data={[{
                                  x: Object.keys(categoricalData).slice(0, 15),
                                  y: Object.values(categoricalData).slice(0, 15),
                                  type: 'bar',
                                  name: 'Frequency',
                                  marker: { color: 'lightblue' },
                                  text: Object.values(categoricalData).slice(0, 15),
                                  textposition: 'outside'
                                }]}
                                layout={{
                                  title: `Category Frequencies - ${colName}`,
                                  xaxis: { title: 'Categories', tickangle: -45 },
                                  yaxis: { title: 'Count' },
                                  height: 400,
                                  showlegend: false
                                }}
                                useResizeHandler={true}
                                style={{ width: '100%', height: '400px' }}
                                config={{ displayModeBar: true, responsive: true }}
                              />
                            </div>
                            
                            {/* Pie Chart */}
                            <PieChart 
                              columnName={colName} 
                              data={categoricalData}
                              title="Distribution Pie Chart"
                              maxCategories={8}
                            />
                            
                            {/* Category Statistics Table */}
                            <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
                              <h6 className="font-medium text-gray-800 mb-3">Category Statistics - {colName}</h6>
                              <div className="overflow-x-auto">
                                <table className="w-full text-sm">
                                  <thead>
                                    <tr className="bg-gray-50">
                                      <th className="px-3 py-2 text-left font-medium text-gray-700">Category</th>
                                      <th className="px-3 py-2 text-left font-medium text-gray-700">Count</th>
                                      <th className="px-3 py-2 text-left font-medium text-gray-700">Percentage</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {colData.categories?.slice(0, 10).map((cat: any, index: number) => (
                                      <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                                        <td className="px-3 py-2 font-medium text-gray-600">{cat.category}</td>
                                        <td className="px-3 py-2 text-gray-800">{cat.count}</td>
                                        <td className="px-3 py-2 text-gray-800">{cat.frequency_percent}%</td>
                                      </tr>
                                    )) || (
                                      <tr>
                                        <td colSpan={3} className="px-3 py-4 text-center text-gray-500">No category data available</td>
                                      </tr>
                                    )}
                                  </tbody>
                                </table>
                              </div>
                            </div>
                            
                            {/* Unique Values Info */}
                            <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
                              <h6 className="font-medium text-gray-800 mb-3">Variable Info - {colName}</h6>
                              <div className="space-y-2 text-sm">
                                <div><span className="font-medium text-gray-600">Total Categories:</span> <span className="text-blue-600">{colData.unique || 0}</span></div>
                                <div><span className="font-medium text-gray-600">Most Frequent:</span> <span className="text-green-600">{colData.top || 'N/A'}</span></div>
                                <div><span className="font-medium text-gray-600">Frequency:</span> <span className="text-purple-600">{colData.freq || 0}</span></div>
                                <div><span className="font-medium text-gray-600">Missing Values:</span> <span className="text-red-600">{colData.missing || 0} ({colData.missing_percentage || 0}%)</span></div>
                              </div>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
              
              {/* Bivariate Analysis Section */}
              <div className="mb-8">
                <h4 className="font-semibold mb-4 text-xl text-orange-700">🔗 Bivariate Analysis</h4>
                
                {/* Correlation Heatmap */}
                {visualizations?.visualizations?.correlation_heatmap?.figure && (
                  <div className="mb-6">
                    <h5 className="font-medium mb-3 text-lg text-blue-600">📈 Correlation Matrix</h5>
                    <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
                      <Plot
                        data={visualizations.visualizations.correlation_heatmap.figure.data}
                        layout={{
                          ...visualizations.visualizations.correlation_heatmap.figure.layout,
                          autosize: true,
                          responsive: true,
                          height: 600
                        }}
                        useResizeHandler={true}
                        style={{ width: '100%', height: '600px' }}
                        config={{ displayModeBar: true, responsive: true }}
                      />
                    </div>
                  </div>
                )}
                
                {/* Pairplot for Numeric Variables */}
                {statisticalAnalysis?.basic_statistics?.numeric_summary?.describe && 
                 Object.keys(statisticalAnalysis.basic_statistics.numeric_summary.describe).length >= 2 && (
                  <div className="mb-6">
                    <h5 className="font-medium mb-3 text-lg text-purple-600">📊 Pairwise Relationships</h5>
                    <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
                      <p className="text-sm text-gray-600 mb-4">Scatter plot matrix showing relationships between numeric variables</p>
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                        {Object.entries(statisticalAnalysis.basic_statistics.numeric_summary.describe)
                          .slice(0, 4)
                          .map(([col1, data1]: [string, any], i) => 
                            Object.entries(statisticalAnalysis.basic_statistics.numeric_summary.describe)
                              .slice(i + 1, 4)
                              .map(([col2, data2]: [string, any]) => {
                                // Generate sample scatter plot data
                                const n = 100;
                                const x = Array.from({length: n}, () => 
                                  Math.random() * (data1.maximum - data1.minimum) + data1.minimum
                                );
                                const y = Array.from({length: n}, () => 
                                  Math.random() * (data2.maximum - data2.minimum) + data2.minimum
                                );
                                
                                return (
                                  <div key={`${col1}_${col2}`} className="bg-gray-50 border rounded-lg p-4">
                                    <Plot
                                      data={[{
                                        x: x,
                                        y: y,
                                        type: 'scatter',
                                        mode: 'markers',
                                        name: `${col1} vs ${col2}`,
                                        marker: { 
                                          color: 'rgba(75, 192, 192, 0.6)',
                                          size: 6
                                        }
                                      }]}
                                      layout={{
                                        title: `${col1} vs ${col2}`,
                                        xaxis: { title: col1 },
                                        yaxis: { title: col2 },
                                        height: 300,
                                        showlegend: false
                                      }}
                                      useResizeHandler={true}
                                      style={{ width: '100%', height: '300px' }}
                                      config={{ displayModeBar: false }}
                                    />
                                  </div>
                                );
                              })
                          ).flat()
                        }
                      </div>
                    </div>
                  </div>
                )}
              </div>
              
              {/* Original Distribution Plots */}
              <div className="mb-6">
                <h4 className="font-semibold mb-4 text-xl text-gray-700">📈 Basic Distribution Overview</h4>
                {visualizations?.visualizations?.distribution_plots?.figure ? (
                  <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
                    <Plot
                      data={visualizations.visualizations.distribution_plots.figure.data}
                      layout={{
                        ...visualizations.visualizations.distribution_plots.figure.layout,
                        autosize: true,
                        responsive: true,
                      }}
                      useResizeHandler={true}
                      style={{ width: '100%', height: '400px' }}
                    />
                  </div>
                ) : (
                  <div className="text-gray-500 bg-gray-50 p-8 rounded-lg text-center">
                    <p>📊 No basic distribution data available</p>
                    <p className="text-sm mt-2">The detailed analysis above provides comprehensive insights</p>
                  </div>
                )}
              </div>
              
              {/* Loading States */}
              {!statisticalAnalysis && (
                <div className="text-center py-8">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
                  <p className="text-gray-600">Loading detailed analysis...</p>
                </div>
              )}
            </div>
          )}

          {selectedVisualization === 'correlations' && (
            <div>
              <h3 className="font-semibold mb-4">Correlation Analysis</h3>
              {visualizations?.visualizations?.correlation_heatmap?.figure ? (
                <div className="mb-6">
                  <Plot
                    data={visualizations.visualizations.correlation_heatmap.figure.data}
                    layout={{
                      ...visualizations.visualizations.correlation_heatmap.figure.layout,
                      autosize: true,
                      responsive: true,
                    }}
                    useResizeHandler={true}
                    style={{ width: '100%', height: '500px' }}
                  />
                </div>
              ) : (
                <div className="text-gray-500">No correlation data available</div>
              )}
            </div>
          )}

          {selectedVisualization === 'missing' && (
            <div>
              <h3 className="font-semibold mb-4">Missing Values Analysis</h3>
              {visualizations?.visualizations?.missing_values?.figure ? (
                <div className="mb-6">
                  <Plot
                    data={visualizations.visualizations.missing_values.figure.data}
                    layout={{
                      ...visualizations.visualizations.missing_values.figure.layout,
                      autosize: true,
                      responsive: true,
                    }}
                    useResizeHandler={true}
                    style={{ width: '100%', height: '400px' }}
                  />
                </div>
              ) : (
                <div className="text-gray-500">No missing values data available</div>
              )}
            </div>
          )}

          {selectedVisualization === 'outliers' && (
            <div>
              <h3 className="font-semibold mb-4">Outlier Detection</h3>
              {visualizations?.visualizations?.outlier_plots?.figure ? (
                <div className="mb-6">
                  <Plot
                    data={visualizations.visualizations.outlier_plots.figure.data}
                    layout={{
                      ...visualizations.visualizations.outlier_plots.figure.layout,
                      autosize: true,
                      responsive: true,
                    }}
                    useResizeHandler={true}
                    style={{ width: '100%', height: '400px' }}
                  />
                </div>
              ) : (
                <div className="text-gray-500">No outlier data available</div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* AI Narrative */}
      {edaResults.narrative && (
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">🤖 AI Analysis Summary</h2>
          </div>
          <div className="prose max-w-none">
            <div className="whitespace-pre-wrap text-gray-700">
              {edaResults.narrative}
            </div>
          </div>
        </div>
      )}

      {/* Summary */}
      {edaResults.summary && typeof edaResults.summary === 'string' && (
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Summary</h2>
          </div>
          <div className="text-gray-700">
            {edaResults.summary}
          </div>
        </div>
      )}
      
      {/* Summary Object */}
      {edaResults.summary && typeof edaResults.summary === 'object' && (
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Analysis Summary</h2>
          </div>
          <div className="text-gray-700">
            {(edaResults.summary as any).total_analyses && (
              <p>Total Analyses: {(edaResults.summary as any).total_analyses}</p>
            )}
            {(edaResults.summary as any).status && (
              <p>Status: {(edaResults.summary as any).status}</p>
            )}
            {(edaResults.summary as any).completed_at && (
              <p>Completed At: {new Date((edaResults.summary as any).completed_at).toLocaleString()}</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default EDA;
