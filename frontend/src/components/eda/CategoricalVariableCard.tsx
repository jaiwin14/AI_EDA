import { useState } from 'react';
import './CategoricalVariableCard.css';

interface CategoricalVariableCardProps {
  columnName: string;
  data: any;
}

const CategoricalVariableCard: React.FC<CategoricalVariableCardProps> = ({ columnName, data }) => {
  const [showDetails, setShowDetails] = useState(false);
  const [activeTab, setActiveTab] = useState('length');

  return (
    <div className="cvc-card">
      <h5 className="cvc-title">{columnName}</h5>
      
      {/* Basic Information Grid */}
      <div className="cvc-grid">
        <div className="cvc-info-box">
          <span className="cvc-label">Distinct:</span>
          <div className="cvc-value">{data?.basic_info?.distinct || 0}</div>
        </div>
        <div className="cvc-info-box">
          <span className="cvc-label">Distinct (%):</span>
          <div className="cvc-value">{data?.basic_info?.distinct_percentage?.toFixed(2) || 0}%</div>
        </div>
        <div className="cvc-info-box">
          <span className="cvc-label">Unique Ratio:</span>
          <div className="cvc-value">{data?.basic_info?.unique_ratio ? (data.basic_info.unique_ratio * 100).toFixed(1) + '%' : 'N/A'}</div>
        </div>
        <div className="cvc-info-box">
          <span className="cvc-label">Missing:</span>
          <div className="cvc-value">{data?.basic_info?.missing || 0}</div>
        </div>
        <div className="cvc-info-box">
          <span className="cvc-label">Missing (%):</span>
          <div className="cvc-value">{data?.basic_info?.missing_percentage?.toFixed(2) || 0}%</div>
        </div>
        <div className="cvc-info-box">
          <span className="cvc-label">Memory Size:</span>
          <div className="cvc-value">{data?.basic_info?.memory_size?.toFixed(2) || 0} KB</div>
        </div>
        <div className="cvc-info-box">
          <span className="cvc-label">Most Frequent:</span>
          <div className="cvc-value">{data?.basic_info?.most_frequent || 'N/A'}</div>
        </div>
        <div className="cvc-info-box">
          <span className="cvc-label">Frequency:</span>
          <div className="cvc-value">{data?.basic_info?.frequency || 0}</div>
        </div>
      </div>

      {/* Show More Details Button */}
      <button
        onClick={() => setShowDetails(!showDetails)}
        className="cvc-btn"
      >
        {showDetails ? 'Hide Details' : 'Show More Details'}
      </button>

      {/* Expandable Details Section */}
      {showDetails && (
        <div className="cvc-details">
          {/* Tab Navigation */}
          <div className="cvc-tabs">
            {['length', 'categories'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`cvc-tab${activeTab === tab ? ' active' : ''}`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          {activeTab === 'length' && (
            <div className="cvc-tab-content">
              <div className="cvc-grid">
                <div className="cvc-info-box">
                  <span className="cvc-label">Max Length:</span>
                  <div className="cvc-value">{data?.length_stats?.max_length || 'N/A'}</div>
                </div>
                <div className="cvc-info-box">
                  <span className="cvc-label">Median Length:</span>
                  <div className="cvc-value">{data?.length_stats?.median_length?.toFixed(2) || 'N/A'}</div>
                </div>
                <div className="cvc-info-box">
                  <span className="cvc-label">Mean Length:</span>
                  <div className="cvc-value">{data?.length_stats?.mean_length?.toFixed(2) || 'N/A'}</div>
                </div>
                <div className="cvc-info-box">
                  <span className="cvc-label">Min Length:</span>
                  <div className="cvc-value">{data?.length_stats?.min_length || 'N/A'}</div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'categories' && (
            <div className="cvc-category-table-container">
              <h6 className="cvc-orange-title">Categories (Frequency Table)</h6>
              {data?.categories && data.categories.length > 0 ? (
                <div style={{ overflowX: 'auto' }}>
                  <table className="cvc-category-table">
                    <thead>
                      <tr>
                        <th>Category</th>
                        <th>Count</th>
                        <th>Frequency (%)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.categories.map((item: any, index: number) => (
                        <tr key={index}>
                          <td>{item.category}</td>
                          <td>{item.count}</td>
                          <td>{item.frequency_percent}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="cvc-no-data">No categories data available</div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default CategoricalVariableCard;
