import React, { useState } from 'react';
import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import Plot from 'react-plotly.js';
import axios from 'axios';

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
            { key: 'distributions', label: 'Distributions' },
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

                  {/* Numerical Columns Detailed Statistics */}
                  {statisticalAnalysis?.basic_statistics?.numeric_summary?.describe && 
                   Object.keys(statisticalAnalysis.basic_statistics.numeric_summary.describe).length > 0 && (
                    <div>
                      <h4 className="font-medium mb-3">Numerical Columns - Descriptive Statistics</h4>
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm border-collapse border border-gray-300">
                          <thead>
                            <tr className="bg-blue-50">
                              <th className="border border-gray-300 px-4 py-3 text-left font-semibold">Statistic</th>
                              {Object.keys(statisticalAnalysis.basic_statistics.numeric_summary.describe).map((col: string) => (
                                <th key={col} className="border border-gray-300 px-4 py-3 text-left font-semibold min-w-[120px]">{col}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'].map((stat: string) => (
                              <tr key={stat} className="hover:bg-gray-50">
                                <td className="border border-gray-300 px-4 py-3 font-medium bg-gray-50">{stat}</td>
                                {Object.keys(statisticalAnalysis.basic_statistics.numeric_summary.describe).map((col: string) => (
                                  <td key={col} className="border border-gray-300 px-4 py-3">
                                    {statisticalAnalysis.basic_statistics.numeric_summary.describe[col]?.[stat] !== undefined && 
                                     statisticalAnalysis.basic_statistics.numeric_summary.describe[col][stat] !== null
                                      ? Number(statisticalAnalysis.basic_statistics.numeric_summary.describe[col][stat]).toFixed(3)
                                      : '-'}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* Categorical Columns Summary */}
                  {statisticalAnalysis?.basic_statistics?.categorical_summary && 
                   Object.keys(statisticalAnalysis.basic_statistics.categorical_summary).length > 0 && (
                    <div>
                      <h4 className="font-medium mb-3">Categorical Columns Summary</h4>
                      <div className="grid gap-6">
                        {Object.entries(statisticalAnalysis.basic_statistics.categorical_summary).map(([col, stats]: [string, any]) => (
                          <div key={col} className="border border-gray-300 rounded-lg p-6 bg-white shadow-sm">
                            <h5 className="font-semibold mb-4 text-lg text-gray-800">{col}</h5>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-sm">
                              <div className="space-y-1">
                                <span className="font-medium text-gray-600">Unique Values:</span>
                                <div className="text-lg font-semibold text-blue-600">{stats?.unique || stats?.unique_count || 'N/A'}</div>
                              </div>
                              <div className="space-y-1">
                                <span className="font-medium text-gray-600">Unique Ratio:</span>
                                <div className="text-lg font-semibold text-green-600">
                                  {stats?.unique_ratio ? (stats.unique_ratio * 100).toFixed(1) + '%' : 'N/A'}
                                </div>
                              </div>
                              <div className="space-y-1">
                                <span className="font-medium text-gray-600">Most Frequent:</span>
                                <div className="text-lg font-semibold text-purple-600 truncate">
                                  {stats?.top || stats?.most_frequent || 'N/A'}
                                </div>
                              </div>
                              <div className="space-y-1">
                                <span className="font-medium text-gray-600">Frequency:</span>
                                <div className="text-lg font-semibold text-orange-600">
                                  {stats?.freq || stats?.most_frequent_count || 'N/A'}
                                </div>
                              </div>
                            </div>
                            {(stats?.value_counts || stats?.top_values) && (
                              <div className="mt-6 pt-4 border-t border-gray-200">
                                <span className="font-medium text-sm text-gray-600 mb-3 block">Top Values:</span>
                                <div className="flex flex-wrap gap-2">
                                  {Object.entries(stats?.value_counts || stats?.top_values || {}).slice(0, 5).map(([value, count]: [string, any]) => (
                                    <span key={value} className="bg-blue-50 border border-blue-200 px-3 py-2 rounded-md text-sm font-medium">
                                      <span className="text-gray-700">{value}:</span> <span className="text-blue-600">{count}</span>
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
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
              <h3 className="font-semibold mb-4">Distribution Analysis</h3>
              {visualizations?.visualizations?.distribution_plots?.figure ? (
                <div className="mb-6">
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
                <div className="text-gray-500">No distribution data available</div>
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
