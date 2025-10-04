import { useState } from 'react';
import './NumericalVariableCard.css';

interface NumericalVariableCardProps {
  columnName: string;
  data: any;
}

const NumericalVariableCard: React.FC<NumericalVariableCardProps> = ({ columnName, data }) => {
  const [showDetails, setShowDetails] = useState(false);
  const [activeTab, setActiveTab] = useState('quantile');

  if (!data) return null;

  return (
    <div className="nvc-card">
      {/* Basic Information */}
      <div className="nvc-main">
        <h4 className="nvc-title">{data.variable_name || columnName}</h4>
        <div className="nvc-grid">
          <div>
            <span className="nvc-label">Data Type:</span>
            <div className="nvc-value">{data.data_type || 'N/A'}</div>
          </div>
          <div>
            <span className="nvc-label">Uniform:</span>
            <div className={`nvc-value ${data.is_uniform ? 'nvc-green' : 'nvc-red'}`}>
              {data.is_uniform ? 'Yes' : 'No'}
            </div>
          </div>
          <div>
            <span className="nvc-label">Unique:</span>
            <div className={`nvc-value ${data.is_unique ? 'nvc-green' : 'nvc-red'}`}>
              {data.is_unique ? 'Yes' : 'No'}
            </div>
          </div>
          <div>
            <span className="nvc-label">Distinct Count:</span>
            <div className="nvc-value">{data.distinct_count?.toLocaleString() || 'N/A'}</div>
          </div>
          <div>
            <span className="nvc-label">Distinct %:</span>
            <div className="nvc-value">{data.distinct_percentage || 0}%</div>
          </div>
          <div>
            <span className="nvc-label">Missing Count:</span>
            <div className="nvc-value">{data.missing_count?.toLocaleString() || 0}</div>
          </div>
          <div>
            <span className="nvc-label">Missing %:</span>
            <div className="nvc-value">{data.missing_percentage || 0}%</div>
          </div>
          <div>
            <span className="nvc-label">Infinite Count:</span>
            <div className="nvc-value">{data.infinite_count?.toLocaleString() || 0}</div>
          </div>
          <div>
            <span className="nvc-label">Infinite %:</span>
            <div className="nvc-value">{data.infinite_percentage || 0}%</div>
          </div>
          <div>
            <span className="nvc-label">Minimum:</span>
            <div className="nvc-value">{data.minimum !== null ? Number(data.minimum).toFixed(3) : 'N/A'}</div>
          </div>
          <div>
            <span className="nvc-label">Maximum:</span>
            <div className="nvc-value">{data.maximum !== null ? Number(data.maximum).toFixed(3) : 'N/A'}</div>
          </div>
          <div>
            <span className="nvc-label">Zeros:</span>
            <div className="nvc-value">{data.zeros_count?.toLocaleString() || 0}</div>
          </div>
          <div>
            <span className="nvc-label">Zeros %:</span>
            <div className="nvc-value">{data.zeros_percentage || 0}%</div>
          </div>
          <div>
            <span className="nvc-label">Negative:</span>
            <div className="nvc-value">{data.negative_count?.toLocaleString() || 0}</div>
          </div>
          <div>
            <span className="nvc-label">Negative %:</span>
            <div className="nvc-value">{data.negative_percentage || 0}%</div>
          </div>
          <div>
            <span className="nvc-label">Memory Size:</span>
            <div className="nvc-value">
              {data.memory_size_bytes ? `${(data.memory_size_bytes / 1024).toFixed(2)} KB` : 'N/A'}
            </div>
          </div>
        </div>
      </div>

      {/* Details Toggle Button */}
      <button
        onClick={() => setShowDetails(!showDetails)}
        className="nvc-btn"
      >
        {showDetails ? 'Hide Details' : 'Show More Details'}
      </button>

      {/* Expandable Details Section */}
      {showDetails && (
        <div className="nvc-details">
          {/* Tab Navigation */}
          <div className="nvc-tabs">
            {['quantile', 'descriptive', 'common', 'extreme'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`nvc-tab${activeTab === tab ? ' active' : ''}`}
              >
                {tab === 'quantile' && 'Quantile Statistics'}
                {tab === 'descriptive' && 'Descriptive Statistics'}
                {tab === 'common' && 'Common Values'}
                {tab === 'extreme' && 'Extreme Values'}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div className="nvc-tab-content">
            {activeTab === 'quantile' && data.quantile_statistics && (
              <div>
                <h5 className="nvc-stat-title">Quantile Statistics</h5>
                <div className="nvc-stat-grid">
                  <div>
                    <span className="nvc-label">Minimum:</span>
                    <div className="nvc-value">{data.quantile_statistics.minimum?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="nvc-label">5th Percentile:</span>
                    <div className="nvc-value">{data.quantile_statistics.percentile_5?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="nvc-label">Q1:</span>
                    <div className="nvc-value">{data.quantile_statistics.q1?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="nvc-label">Median:</span>
                    <div className="nvc-value">{data.quantile_statistics.median?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="nvc-label">Q3:</span>
                    <div className="nvc-value">{data.quantile_statistics.q3?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="nvc-label">95th Percentile:</span>
                    <div className="nvc-value">{data.quantile_statistics.percentile_95?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="nvc-label">Maximum:</span>
                    <div className="nvc-value">{data.quantile_statistics.maximum?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="nvc-label">Range:</span>
                    <div className="nvc-value">{data.quantile_statistics.range?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="nvc-label">IQR:</span>
                    <div className="nvc-value">{data.quantile_statistics.iqr?.toFixed(3) || 'N/A'}</div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'descriptive' && data.descriptive_statistics && (
              <div>
                <h5 className="nvc-stat-title">Descriptive Statistics</h5>
                <div className="nvc-stat-grid">
                  <div>
                    <span className="nvc-label">Standard Deviation:</span>
                    <div className="nvc-value">{data.descriptive_statistics.standard_deviation?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="nvc-label">CV:</span>
                    <div className="nvc-value">{data.descriptive_statistics.coefficient_of_variation || 0}%</div>
                  </div>
                  <div>
                    <span className="nvc-label">Kurtosis:</span>
                    <div className="nvc-value">{data.descriptive_statistics.kurtosis?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="nvc-label">Mean:</span>
                    <div className="nvc-value">{data.descriptive_statistics.mean?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="nvc-label">MAD:</span>
                    <div className="nvc-value">{data.descriptive_statistics.median_absolute_deviation?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="nvc-label">Skewness:</span>
                    <div className="nvc-value">{data.descriptive_statistics.skewness?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="nvc-label">Sum:</span>
                    <div className="nvc-value">{data.descriptive_statistics.sum?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="nvc-label">Variance:</span>
                    <div className="nvc-value">{data.descriptive_statistics.variance?.toFixed(3) || 'N/A'}</div>
                  </div>
                  <div>
                    <span className="nvc-label">Monotonicity:</span>
                    <div className="nvc-value">{data.descriptive_statistics.monotonicity || 'N/A'}</div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'common' && data.common_values && (
              <div>
                <h5 className="nvc-stat-title">Most Frequently Occurring Values</h5>
                <div className="nvc-common-list">
                  {data.common_values.values?.length > 0 ? (
                    data.common_values.values.map((item: any, index: number) => (
                      <div key={index} className="nvc-common-item">
                        <span className="nvc-value">{item.value?.toFixed(3)}</span>
                        <span className="nvc-gray">
                          Count: {item.count?.toLocaleString()} ({item.percentage}%)
                        </span>
                      </div>
                    ))
                  ) : (
                    <div className="nvc-no-data">No common values data available</div>
                  )}
                </div>
              </div>
            )}

            {activeTab === 'extreme' && data.extreme_values && (
              <div>
                <h5 className="nvc-stat-title">Extreme Values</h5>
                <div className="nvc-extreme-list">
                  <div>
                    <h6 className="nvc-label nvc-red">Lowest Values</h6>
                    {data.extreme_values.lowest?.length > 0 ? (
                      data.extreme_values.lowest.map((item: any, index: number) => (
                        <div key={index} className="nvc-extreme-item nvc-extreme-low">
                          <span className="nvc-value">{item.value?.toFixed(3)}</span>
                          <span className="nvc-gray">Index: {item.index}</span>
                        </div>
                      ))
                    ) : (
                      <div className="nvc-no-data">No data available</div>
                    )}
                  </div>
                  <div>
                    <h6 className="nvc-label nvc-green">Highest Values</h6>
                    {data.extreme_values.highest?.length > 0 ? (
                      data.extreme_values.highest.map((item: any, index: number) => (
                        <div key={index} className="nvc-extreme-item nvc-extreme-high">
                          <span className="nvc-value">{item.value?.toFixed(3)}</span>
                          <span className="nvc-gray">Index: {item.index}</span>
                        </div>
                      ))
                    ) : (
                      <div className="nvc-no-data">No data available</div>
                    )}
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
