import { useState } from 'react';
import OverviewTab from './OverviewTab';
import StatisticsTab from './StatisticsTab';
import DistributionsTab from './DistributionsTab';
import CorrelationInsights from './CorrelationInsights';
import MissingValuesInsights from './MissingValuesInsights';
import OutlierAnalysisSection from './OutlierAnalysisSection';
import './EDALayout.css';

interface EDALayoutProps {
  datasetId: string;
  edaResults: any;
  statisticalAnalysis: any;
  visualizations: any;
}

const EDALayout: React.FC<EDALayoutProps> = ({ 
  datasetId, 
  edaResults, 
  statisticalAnalysis, 
  visualizations 
}) => {
  const [selectedVisualization, setSelectedVisualization] = useState<string>('overview');

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

      {/* Dataset Overview Stats */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Dataset Overview</h2>
        </div>
        <div className="tables-center">
          <div className="table-container">
            <table className="main-table">
              <thead>
                <tr>
                  <th className="th-cell">Metric</th>
                  <th className="th-cell">Value</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="td-label">Variables/Attributes</td>
                  <td className="td-value">
                    {statisticalAnalysis?.dataset_overview?.total_variables?.toLocaleString() || 
                     statisticalAnalysis?.dataset_overview?.total_columns?.toLocaleString() || 
                     basicStats.total_columns?.toLocaleString() || 'N/A'}
                  </td>
                </tr>
                
                
                <tr>
                  <td className="td-label">Duplicate Rows</td>
                  <td className="td-value">
                    {statisticalAnalysis?.dataset_overview?.duplicate_rows?.toLocaleString() || 'N/A'}
                  </td>
                </tr>
                <tr>
                  <td className="td-label">Duplicate Rows (%)</td>
                  <td className="td-value">
                    {statisticalAnalysis?.dataset_overview?.duplicate_rows_percentage !== undefined 
                      ? `${statisticalAnalysis.dataset_overview.duplicate_rows_percentage.toFixed(1)}%`
                      : 'N/A'}
                  </td>
                </tr>
                <tr>
                  <td className="td-label">Total Memory Size</td>
                  <td className="td-value">
                    {statisticalAnalysis?.dataset_overview?.total_memory_usage_mb !== undefined 
                      ? `${statisticalAnalysis.dataset_overview.total_memory_usage_mb.toFixed(2)} MB`
                      : 'N/A'}
                  </td>
                </tr>
                <tr>
                  <td className="td-label">Avg Record Size</td>
                  <td className="td-value">
                    {statisticalAnalysis?.dataset_overview?.average_record_size_bytes !== undefined 
                      ? `${(statisticalAnalysis.dataset_overview.average_record_size_bytes / 1024).toFixed(2)} KB`
                      : 'N/A'}
                  </td>
                </tr>
                <tr>
                  <td className="td-label">Data Records</td>
                  <td className="td-value">
                    {statisticalAnalysis?.dataset_overview?.number_of_data_records?.toLocaleString() || 
                     statisticalAnalysis?.dataset_overview?.total_rows?.toLocaleString() || 
                     basicStats.total_rows?.toLocaleString() || 'N/A'}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Variable Types Section */}
          <div className="table-container">
            <h3 className="section-title">Variable Types</h3>
            <table className="main-table">
              <thead>
                <tr>
                  <th className="th-cell">Type</th>
                  <th className="th-cell">Count</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="td-label">Numeric Variables</td>
                  <td className="td-value">
                    {statisticalAnalysis?.dataset_overview?.variable_types?.numeric || 
                     statisticalAnalysis?.dataset_overview?.numeric_columns || 
                     basicStats.numeric_columns || 'N/A'}
                  </td>
                </tr>
                <tr>
                  <td className="td-label">Categorical Variables</td>
                  <td className="td-value">
                    {statisticalAnalysis?.dataset_overview?.variable_types?.categorical || 
                     statisticalAnalysis?.dataset_overview?.categorical_columns || 
                     basicStats.categorical_columns || 'N/A'}
                  </td>
                </tr>
                <tr>
                  <td className="td-label">DateTime Variables</td>
                  <td className="td-value">
                    {statisticalAnalysis?.ataset_overview?.variable_types?.datetime || 
                     statisticalAnalysis?.dataset_overview?.datetime_columns || 0}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="card">
        <div className="tab-row">
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
              className={`tab-btn ${selectedVisualization === tab.key ? 'active' : ''}`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div>
          {selectedVisualization === 'overview' && (
            <OverviewTab 
              datasetId={datasetId}
              basicStats={basicStats}
              missingValues={missingValues}
            />
          )}

          {selectedVisualization === 'statistics' && (
            <StatisticsTab 
              datasetId={datasetId}
              statisticalAnalysis={statisticalAnalysis}
            />
          )}

          {selectedVisualization === 'distributions' && (
            <DistributionsTab 
              datasetId={datasetId}
              statisticalAnalysis={statisticalAnalysis}
              visualizations={visualizations}
            />
          )}

          {selectedVisualization === 'correlations' && (
            <CorrelationInsights 
              datasetId={datasetId} 
              correlationData={visualizations?.visualizations?.correlation_heatmap}
            />
          )}

          {selectedVisualization === 'missing' && (
            <MissingValuesInsights datasetId={datasetId} />
          )}

          {selectedVisualization === 'outliers' && (
            <OutlierAnalysisSection datasetId={datasetId} />
          )}
        </div>
      </div>
    </div>
  );
};

export default EDALayout;
