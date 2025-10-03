import { useState } from 'react';
import OverviewTab from './OverviewTab';
import StatisticsTab from './StatisticsTab';
import DistributionsTab from './DistributionsTab';
import CorrelationInsights from './CorrelationInsights';
import MissingValuesInsights from './MissingValuesInsights';
import OutlierAnalysisSection from './OutlierAnalysisSection';

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

        {/* Tab Content */}
        <div className="mt-4">
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
