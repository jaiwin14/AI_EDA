import { useQuery } from '@tanstack/react-query';
import aiService from '../../services/aiService';
import './OverviewTab.css';

interface OverviewTabProps {
  datasetId: string;
  basicStats: any;
  missingValues: any;
}

const OverviewTab: React.FC<OverviewTabProps> = ({ datasetId, basicStats, missingValues }) => {
  const { data: aiOverview, isLoading: aiOverviewLoading, error: aiOverviewError } = useQuery({
    queryKey: ['ai-overview', datasetId],
    queryFn: async () => {
      if (!datasetId) throw new Error('Dataset ID is required');
      const result = await aiService.getDatasetOverview(datasetId);
      return result;
    },
    enabled: !!datasetId,
    retry: 1,
  });

  return (
    <div>
      {/* AI Overview Section */}
      {aiOverviewLoading && (
        <div className="section loading">
          <div className="spinner"></div>
        </div>
      )}

      {aiOverviewError && (
        <div className="section error">
          <p className="section-title error">Failed to load AI overview: {String(aiOverviewError)}</p>
        </div>
      )}

      {/* Render basic dataset overview */}
      {aiOverview?.data && (
        <div className="section">
          <h3 className="title">Dataset Overview</h3>
          <div className="overview-grid">
            <div className="overview-item">
              <span className="overview-label">Shape:</span>
              <span className="overview-value">
                {aiOverview.data.shape?.[0] || 0} rows × {aiOverview.data.shape?.[1] || 0} columns
              </span>
            </div>
            <div className="overview-item">
              <span className="overview-label">Duplicate Rows:</span>
              <span className="overview-value">{aiOverview.data.duplicate_rows || 0}</span>
            </div>
            <div className="overview-item">
              <span className="overview-label">Memory Usage:</span>
              <span className="overview-value">{aiOverview.data.memory_usage || '0 MB'}</span>
            </div>
          </div>

          {/* Columns List */}
          <div className="section">
            <h4 className="title">Columns ({aiOverview.data.columns?.length || 0})</h4>
            <div className="columns-grid">
              {aiOverview.data.columns?.map((column: string, index: number) => (
                <div key={index} className="column-card">
                  <span className="column-name">{column}</span>
                  <span className="column-index">#{index + 1}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Data Types Table */}
          <div className="section">
            <h4 className="title">Data Types</h4>
            <div className="table-responsive">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Column</th>
                    <th>Data Type</th>
                  </tr>
                </thead>
                <tbody>
                  {aiOverview.data.dtypes &&
                    Object.entries(aiOverview.data.dtypes).map(([column, dtype]) => (
                      <tr key={column}>
                        <td>{column}</td>
                        <td>
                          <span className={`dtype-badge dtype-${dtype}`}>
                            {String(dtype)}
                          </span>
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
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
          <div className="section">
            <h3 className="title">
              AI Dataset Overview
            </h3>
            <div>
              <p>{overviewData.overview_line_1}</p>
              <p>{overviewData.overview_line_2}</p>
              {overviewData.key_characteristics && (
                <div className="key-characteristics">
                  <h4 className="key-characteristics-title">Key Characteristics:</h4>
                  <ul className="key-list">
                    {overviewData.key_characteristics.map((characteristic: string, index: number) => (
                      <li key={index} className="key-list-item">{characteristic}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        );
      })()}

      <h3 className="data-quality-title">Data Quality Summary</h3>
      {missingValues.columns_with_missing && 
       Object.keys(missingValues.columns_with_missing).length > 0 ? (
        <div>
          <h4 className="data-quality-title">Columns with Missing Values</h4>
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Column</th>
                  <th>Missing Count</th>
                  <th>Missing %</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(missingValues.columns_with_missing).map(([col, count]) => (
                  <tr key={col}>
                    <td>{col}</td>
                    <td>{count as number}</td>
                    <td>
                      {((count as number / basicStats.total_rows) * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="success">No missing values detected!</div>
      )}
    </div>
  );
};

export default OverviewTab;