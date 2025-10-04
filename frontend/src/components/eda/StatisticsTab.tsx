import { useQuery } from '@tanstack/react-query';
import aiService from '../../services/aiService';
import NumericalVariableCard from './NumericalVariableCard';
import CategoricalVariableCard from './CategoricalVariableCard';
import './StatisticsTab.css';

// Helper function to render objects as nested divs
function renderObjectAsDiv(obj: any, level: number = 0): JSX.Element {
  console.log('AI Summary:', obj);
  if (obj === null || obj === undefined) return <div className="stat-raw-row">null</div>;
  if (typeof obj !== 'object') return <div className="stat-raw-row">{String(obj)}</div>;
  if (Array.isArray(obj)) {
    return (
      <div className="stat-raw-array" style={{ marginLeft: level * 16 }}>
        {obj.map((item, idx) => (
          <div key={idx}>{renderObjectAsDiv(item, level + 1)}</div>
        ))}
      </div>
    );
  }
  return (
    <div className="stat-raw-object" style={{ marginLeft: level * 16 }}>
      {Object.entries(obj).map(([key, value]) => (
        <div key={key} className="stat-raw-col">
          <span className="stat-raw-key">{key}:</span>
          <span className="stat-raw-value">{renderObjectAsDiv(value, level + 1)}</span>
        </div>
      ))}
    </div>
  );
}

interface StatisticsTabProps {
  datasetId: string;
  statisticalAnalysis: any;
}

const StatisticsTab: React.FC<StatisticsTabProps> = ({ datasetId, statisticalAnalysis }) => {
  const { data: aiSummary, isLoading: aiSummaryLoading, error: aiSummaryError } = useQuery({
    queryKey: ['ai-summary', datasetId],
    queryFn: async () => {
      if (!datasetId) throw new Error('Dataset ID is required');
      const result = await aiService.getStatisticalInsights(datasetId);
      return result;
    },
    enabled: !!datasetId,
    retry: 1,
  });

  return (
    <div>
      {/* AI Summary Section */}
      {aiSummaryLoading && (
        <div className="ai-section ai-loading">
          <div className="ai-spinner"></div>
          <span>Loading AI statistical insights...</span>
        </div>
      )}
      
      {aiSummaryError && (
        <div className="ai-section error">
          <p className="ai-section-title error">Failed to load AI summary: {String(aiSummaryError)}</p>
        </div>
      )}
      
      {(() => {
        let summaryPoints = null;
        if (aiSummary?.data?.summary?.summary_points) {
          summaryPoints = aiSummary.data.summary.summary_points;
        } else if (aiSummary?.data?.insights) {
          try {
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
          <div className="ai-section">
            <h3 className="ai-section-title">
              AI Statistical Insights (10 Key Points)
            </h3>
            <ol className="ai-key-list">
              {summaryPoints.map((point: string, index: number) => (
                <li key={index} className="ai-key-list-item">
                  {point}
                </li>
              ))}
            </ol>
          </div>
        );
      })()}
      
      {aiSummary && !aiSummary?.data?.summary?.summary_points && (
        <div className="ai-section raw">
          <h3 className="ai-section-title raw">
            AI Statistical Summary (Raw)
          </h3>
          <pre className="stat-raw-col">
            Success: {renderObjectAsDiv(aiSummary?.success)}
          </pre>
          <pre className="stat-raw-col">
            Analysis Type: {renderObjectAsDiv(aiSummary?.analysis_type)}
          </pre>
          <pre className="stat-raw-col">
            Dataset ID: {renderObjectAsDiv(aiSummary?.dataset_id)}
          </pre>
          <pre className="stat-raw-col">
            Timestamp: {aiSummary?.timestamp ? new Date(aiSummary.timestamp).toLocaleString() : ''}
          </pre>
          <pre className="stat-raw-data">
            Summary points: {renderObjectAsDiv(aiSummary?.data?.insights)}
          </pre>
        </div>
      )}
      
      <h3 className="stat-title">Statistical Analysis</h3>
      
      {statisticalAnalysis ? (
        <div>
          {/* Dataset Overview */}
          <div className="stat-overview">
            <h4 className="stat-title-1">Dataset Overview</h4>
            <div className="stat-overview-grid">
              <div className="stat-overview-item">
                <span className="stat-label">Shape:</span>
                <span className="stat-value">
                  {statisticalAnalysis?.dataset_shape ? `${statisticalAnalysis.dataset_shape[0]?.toLocaleString()} × ${statisticalAnalysis.dataset_shape[1]}` : 'N/A'}
                </span>
              </div>
              <div className="stat-overview-item">
                <span className="stat-label">Total Columns:</span>
                <span className="stat-value">
                  {statisticalAnalysis?.dataset_overview?.total_columns || 0}
                </span>
              </div>
              <div className="stat-overview-item">
                <span className="stat-label">Numeric Columns:</span>
                <span className="stat-value">
                  {statisticalAnalysis?.dataset_overview?.numeric_columns || 0}
                </span>
              </div>
              <div className="stat-overview-item">
                <span className="stat-label">Categorical Columns:</span>
                <span className="stat-value">
                  {statisticalAnalysis?.dataset_overview?.categorical_columns || 0}
                </span>
              </div>
              <div className="stat-overview-item">
                <span className="stat-label">Memory Usage:</span>
                <span className="stat-value">
                  {statisticalAnalysis?.dataset_overview?.memory_usage_mb 
                    ? `${statisticalAnalysis.dataset_overview.memory_usage_mb.toFixed(2)} MB` 
                    : 'N/A'}
                </span>
              </div>
              <div className="stat-overview-item">
                <span className="stat-label">Missing Values:</span>
                <span className="stat-value">
                  {statisticalAnalysis?.dataset_overview?.missing_percentage !== undefined 
                    ? `${statisticalAnalysis.dataset_overview.missing_percentage.toFixed(2)}%` 
                    : '0%'}
                </span>
              </div>
              <div className="stat-overview-item">
                <span className="stat-label">Duplicate Rows:</span>
                <span className="stat-value">
                  {statisticalAnalysis?.dataset_overview?.duplicate_rows?.toLocaleString() || 0}
                </span>
              </div>
              <div className="stat-overview-item">
                <span className="stat-label">DateTime Columns:</span>
                <span className="stat-value">
                  {statisticalAnalysis?.dataset_overview?.datetime_columns || 0}
                </span>
              </div>
            </div>
          </div>

          {/* Column Summary Table */}
          <div className="stat-table-container">
            <h4 className="stat-title">Column Summary</h4>
            <table className="stat-table">
              <thead>
                <tr>
                  <th>Column</th>
                  <th>Type</th>
                  <th>Data Type</th>
                  <th>Non-Null</th>
                  <th>Missing %</th>
                  <th>Unique</th>
                  <th>Min</th>
                  <th>Q1</th>
                  <th>Median</th>
                  <th>Q3</th>
                  <th>Max</th>
                  <th>Mean</th>
                  <th>Std Dev</th>
                </tr>
              </thead>
              <tbody>
                {statisticalAnalysis?.column_summary?.map((col: any, index: number) => (
                  <tr key={index}>
                    <td>{col?.column_name || 'N/A'}</td>
                    <td>
                      <span className={`type-badge ${col?.column_type === 'numerical' ? 'numerical' : 'categorical'}`}>
                        {col?.column_type || 'unknown'}
                      </span>
                    </td>
                    <td>{col?.data_type || 'N/A'}</td>
                    <td>{col?.non_null_count?.toLocaleString() || 'N/A'}</td>
                    <td>{col?.null_percentage !== undefined ? `${col.null_percentage.toFixed(1)}%` : 'N/A'}</td>
                    <td>{col?.unique_count?.toLocaleString() || 'N/A'}</td>
                    <td>{col?.column_type === 'numerical' && col?.min !== null && col?.min !== undefined ? Number(col.min).toFixed(3) : '-'}</td>
                    <td>{col?.column_type === 'numerical' && col?.q25 !== null && col?.q25 !== undefined ? Number(col.q25).toFixed(3) : '-'}</td>
                    <td>{col?.column_type === 'numerical' && col?.median !== null && col?.median !== undefined ? Number(col.median).toFixed(3) : '-'}</td>
                    <td>{col?.column_type === 'numerical' && col?.q75 !== null && col?.q75 !== undefined ? Number(col.q75).toFixed(3) : '-'}</td>
                    <td>{col?.column_type === 'numerical' && col?.max !== null && col?.max !== undefined ? Number(col.max).toFixed(3) : '-'}</td>
                    <td>{col?.column_type === 'numerical' && col?.mean !== null && col?.mean !== undefined ? Number(col.mean).toFixed(3) : '-'}</td>
                    <td>{col?.column_type === 'numerical' && col?.std !== null && col?.std !== undefined ? Number(col.std).toFixed(3) : '-'}</td>
                  </tr>
                )) || (
                  <tr>
                    <td colSpan={13} className="stat-empty">
                      No column data available
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          {/* Numerical Variables Analysis */}
          {statisticalAnalysis?.basic_statistics?.numeric_summary?.describe && 
           Object.keys(statisticalAnalysis.basic_statistics.numeric_summary.describe).length > 0 && (
            <div className="stat-section">
              <h4 className="stat-title">Numerical Variables Analysis</h4>
              <div className="stat-card-list">
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
            <div className="stat-section">
              <h4 className="stat-title">Categorical Variables Analysis</h4>
              <div className="stat-card-list">
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
        <div className="stat-loading">Loading statistical analysis...</div>
      )}
    </div>
  );
};

export default StatisticsTab;
