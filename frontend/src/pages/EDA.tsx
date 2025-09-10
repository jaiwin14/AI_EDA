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
  summary: string;
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
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <div className="text-2xl font-bold text-blue-600">
              {basicStats.total_rows?.toLocaleString() || 'N/A'}
            </div>
            <div className="text-sm text-gray-600">Total Rows</div>
          </div>
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <div className="text-2xl font-bold text-green-600">
              {basicStats.total_columns || 'N/A'}
            </div>
            <div className="text-sm text-gray-600">Total Columns</div>
          </div>
          <div className="text-center p-4 bg-purple-50 rounded-lg">
            <div className="text-2xl font-bold text-purple-600">
              {basicStats.numeric_columns || 'N/A'}
            </div>
            <div className="text-sm text-gray-600">Numeric Columns</div>
          </div>
          <div className="text-center p-4 bg-orange-50 rounded-lg">
            <div className="text-2xl font-bold text-orange-600">
              {missingValues.overall_missing_percentage?.toFixed(1) || '0'}%
            </div>
            <div className="text-sm text-gray-600">Missing Values</div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="card">
        <div className="flex flex-wrap gap-2 mb-4">
          {[
            { key: 'overview', label: 'Overview' },
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

          {selectedVisualization === 'distributions' && (
            <div>
              <h3 className="font-semibold mb-4">Distribution Analysis</h3>
              {visualizations?.distributions?.map((viz: any, index: number) => (
                <div key={index} className="mb-6">
                  <Plot
                    data={viz.data}
                    layout={{
                      ...viz.layout,
                      autosize: true,
                      responsive: true,
                    }}
                    useResizeHandler={true}
                    style={{ width: '100%', height: '400px' }}
                  />
                </div>
              ))}
            </div>
          )}

          {selectedVisualization === 'correlations' && (
            <div>
              <h3 className="font-semibold mb-4">Correlation Analysis</h3>
              {visualizations?.correlations?.map((viz: any, index: number) => (
                <div key={index} className="mb-6">
                  <Plot
                    data={viz.data}
                    layout={{
                      ...viz.layout,
                      autosize: true,
                      responsive: true,
                    }}
                    useResizeHandler={true}
                    style={{ width: '100%', height: '500px' }}
                  />
                </div>
              ))}
            </div>
          )}

          {selectedVisualization === 'missing' && (
            <div>
              <h3 className="font-semibold mb-4">Missing Values Analysis</h3>
              {visualizations?.missing_values?.map((viz: any, index: number) => (
                <div key={index} className="mb-6">
                  <Plot
                    data={viz.data}
                    layout={{
                      ...viz.layout,
                      autosize: true,
                      responsive: true,
                    }}
                    useResizeHandler={true}
                    style={{ width: '100%', height: '400px' }}
                  />
                </div>
              ))}
            </div>
          )}

          {selectedVisualization === 'outliers' && (
            <div>
              <h3 className="font-semibold mb-4">Outlier Detection</h3>
              {visualizations?.outliers?.map((viz: any, index: number) => (
                <div key={index} className="mb-6">
                  <Plot
                    data={viz.data}
                    layout={{
                      ...viz.layout,
                      autosize: true,
                      responsive: true,
                    }}
                    useResizeHandler={true}
                    style={{ width: '100%', height: '400px' }}
                  />
                </div>
              ))}
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
      {edaResults.summary && (
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Summary</h2>
          </div>
          <div className="text-gray-700">
            {edaResults.summary}
          </div>
        </div>
      )}
    </div>
  );
};

export default EDA;
