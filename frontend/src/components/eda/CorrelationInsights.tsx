import React, { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import Plot from 'react-plotly.js';
import aiService from '../../services/aiService';
import './CorrelationInsights.css';


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

interface CorrelationData {
  var1: string;
  var2: string;
  correlation: number;
}

interface CorrelationInsightsProps {
  datasetId: string;
  correlationData?: any;
}

const CorrelationInsights: React.FC<CorrelationInsightsProps> = ({ 
  datasetId, 
  correlationData: propCorrelationData 
}) => {
  const [activeTab, setActiveTab] = useState<'heatmap' | 'insights' | 'recommendations'>('heatmap');

  // Fetch correlation insights
  const { data: correlationData, isLoading: aiLoading, error: aiError } = useQuery({
    queryKey: ['correlation-insights', datasetId],
    queryFn: () => aiService.getCorrelationInsights(datasetId),
    enabled: !!datasetId,
    retry: 2,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  const correlationInsights = correlationData?.data;
  const fallbackInsights: string[] = [];
  const allCorrelations: CorrelationData[] = correlationInsights?.correlations || [];
  const correlationMatrix = correlationInsights?.correlation_matrix || {};
  const visualizationData = propCorrelationData;

  // Debug logging
  React.useEffect(() => {
    if (correlationData) {
      console.log('🔍 Correlation Debug - Correlation Data:', correlationData);
      console.log('📊 Correlations found:', allCorrelations.length);
      console.log('💡 Insights found:', fallbackInsights.length);
      console.log('🔗 Matrix keys:', Object.keys(correlationMatrix).length);
    }
  }, [correlationData, allCorrelations, fallbackInsights, correlationMatrix]);

  // Process correlation data for better display
  const processedCorrelations = useMemo(() => {
    if (!allCorrelations.length) return { strong: [], moderate: [], weak: [] };
    
    const strong = allCorrelations.filter((c: CorrelationData) => Math.abs(c.correlation) >= 0.7);
    const moderate = allCorrelations.filter((c: CorrelationData) => Math.abs(c.correlation) >= 0.3 && Math.abs(c.correlation) < 0.7);
    const weak = allCorrelations.filter((c: CorrelationData) => Math.abs(c.correlation) < 0.3);
    
    return { strong, moderate, weak };
  }, [allCorrelations]);

  // Generate insights based on correlation data
  const generateInsights = useMemo(() => {
    if (!allCorrelations.length) return [];
    
    const insights = [];
    const totalVars = Object.keys(correlationMatrix).length;
    
    // Basic statistics
    insights.push(`Analyzed ${totalVars} numeric variables with ${allCorrelations.length} correlation pairs`);
    
    // Strong correlations
    if (processedCorrelations.strong.length > 0) {
      insights.push(`Found ${processedCorrelations.strong.length} strong correlations (|r| ≥ 0.7)`);
      const strongestCorr = processedCorrelations.strong[0];
      const direction = strongestCorr.correlation > 0 ? 'positive' : 'negative';
      insights.push(`Strongest ${direction} correlation: ${strongestCorr.var1} ↔ ${strongestCorr.var2} (r = ${strongestCorr.correlation.toFixed(3)})`);
    }
    
    // Moderate correlations
    if (processedCorrelations.moderate.length > 0) {
      insights.push(`Found ${processedCorrelations.moderate.length} moderate correlations (0.3 <= |r| < 0.7)`);
    }
    
    // Multicollinearity warning
    if (processedCorrelations.strong.length > 0) {
      insights.push(`Strong correlations detected - consider checking for multicollinearity in modeling`);
    }
    
    // Data quality insights
    const positiveCorrs = allCorrelations.filter((c: CorrelationData) => c.correlation > 0).length;
    const negativeCorrs = allCorrelations.filter((c: CorrelationData) => c.correlation < 0).length;
    insights.push(`Distribution: ${positiveCorrs} positive, ${negativeCorrs} negative correlations`);
    
    return insights;
  }, [allCorrelations, correlationMatrix, processedCorrelations]);

  // Generate recommendations
  const generateRecommendations = useMemo(() => {
    if (!allCorrelations.length) return [];
    
    const recommendations = [];
    
    // Strong correlation recommendations
    if (processedCorrelations.strong.length > 0) {
      recommendations.push('Investigate strong correlations for potential feature redundancy');
      recommendations.push('Consider removing highly correlated features to reduce multicollinearity');
      recommendations.push('Use techniques like PCA or feature selection to handle correlated features');
    }
    
    // Moderate correlation recommendations
    if (processedCorrelations.moderate.length > 0) {
      recommendations.push('Moderate correlations may indicate meaningful relationships worth exploring');
      recommendations.push('Consider domain knowledge to interpret moderate correlations');
    }
    
    // General recommendations
    recommendations.push('Remember: correlation does not imply causation');
    recommendations.push('Focus on correlations relevant to your target variable for predictive modeling');
    recommendations.push('Document significant correlations for stakeholder communication');
    
    return recommendations;
  }, [allCorrelations, processedCorrelations]);

  return (
    <div className="corr-container">
      {/* Header */}
      <div>
        <h3 className="corr-title">Correlation Analysis</h3>
        <p className="corr-desc">
          Explore relationships between numeric variables with AI-powered insights
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="corr-tabs">
        <nav className="corr-tab-nav" aria-label="Tabs">
          {[
            { key: 'heatmap', label: 'Correlation Matrix' },
            { key: 'insights', label: 'AI Insights' },
            { key: 'recommendations', label: 'Recommendations' }
          ].map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key as any)}
              className={`corr-tab-btn${activeTab === tab.key ? ' active' : ''}`}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        <div className="corr-tab-content">
          {/* Correlation Heatmap Tab */}
          {activeTab === 'heatmap' && (
            <div className="corr-heatmap-section">
              {visualizationData?.figure ? (
                <>
                  <div className="corr-heatmap-plot">
                    <Plot
                      data={visualizationData.figure.data}
                      layout={{
                        ...visualizationData.figure.layout,
                        autosize: true,
                        responsive: true,
                        height: 600,
                        margin: { l: 100, r: 50, t: 50, b: 100 }
                      }}
                      config={{
                        displayModeBar: true,
                        displaylogo: false,
                        modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
                      }}
                      style={{ width: '100%', height: '600px' }}
                    />
                  </div>
                  <div className="corr-stats-grid">
                    <div className="corr-stat-box corr-stat-strong">
                      <div>{processedCorrelations.strong.length}</div>
                      <div>Strong Correlations</div>
                      <div>|r| ≥ 0.7</div>
                    </div>
                    <div className="corr-stat-box corr-stat-moderate">
                      <div>{processedCorrelations.moderate.length}</div>
                      <div>Moderate Correlations</div>
                      <div>0.3 ≤ |r| &lt; 0.7</div>
                    </div>
                    <div className="corr-stat-box corr-stat-weak">
                      <div>{processedCorrelations.weak.length}</div>
                      <div>Weak Correlations</div>
                      <div>|r| &lt; 0.3</div>
                    </div>
                    <div className="corr-stat-box corr-stat-vars">
                      <div>{Object.keys(correlationMatrix).length}</div>
                      <div>Variables Analyzed</div>
                      <div>Numeric only</div>
                    </div>
                  </div>
                  {allCorrelations.length > 0 && (
                    <div className="corr-top-list">
                      <h4 className="corr-top-label">Top Correlations</h4>
                      {allCorrelations.slice(0, 10).map((corr: CorrelationData, index: number) => {
                        const strength = Math.abs(corr.correlation) >= 0.7 ? 'strong' : 
                                       Math.abs(corr.correlation) >= 0.3 ? 'moderate' : 'weak';
                        const strengthClass = strength === 'strong' ? 'corr-strength-strong' :
                                             strength === 'moderate' ? 'corr-strength-moderate' :
                                             'corr-strength-weak';
                        return (
                          <div key={index} className="corr-top-item">
                            <div>
                              <span className="corr-top-label">{corr.var1} ↔ {corr.var2}</span>
                            </div>
                            <div>
                              <span className={`corr-top-strength ${strengthClass}`}>{strength}</span>
                              <span className="corr-strength-value">{corr.correlation.toFixed(3)}</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </>
              ) : (
                <div className="corr-no-data">
                  <p>No correlation heatmap available</p>
                  <p style={{fontSize: '0.95rem', marginTop: '0.5rem'}}>Ensure your dataset has at least 2 numeric columns</p>
                </div>
              )}
            </div>
          )}

          {/* AI Insights Tab */}
          {activeTab === 'insights' && (
            <div className="corr-insights-section">
              {aiLoading ? (
                <div className="corr-no-data">
                  <div className="corr-spinner"></div>
                  <p>Generating AI insights...</p>
                </div>
              ) : aiError ? (
                <div className="corr-insight-box" style={{background: "#fee2e2", borderColor: "#fecaca"}}>
                  <div>
                    <h4 className="corr-insight-title" style={{color: "#dc2626"}}>Error Loading AI Insights</h4>
                    <p>Unable to generate correlation insights. Please try again.</p>
                    <details>
                      <summary>Debug Info</summary>
                      <pre>{JSON.stringify(aiError, null, 2)}</pre>
                    </details>
                  </div>
                </div>
              ) : !correlationData ? (
                <div className="corr-insight-box" style={{background: "#fef9c3", borderColor: "#fde68a"}}>
                  <div>
                    <h4 className="corr-insight-title" style={{color: "#ca8a04"}}>No Data Available</h4>
                    <p>No correlation data received from the server.</p>
                  </div>
                </div>
              ) : allCorrelations.length === 0 ? (
                <div className="corr-insight-box" style={{background: "#fff7ed", borderColor: "#fed7aa"}}>
                  <div>
                    <h4 className="corr-insight-title" style={{color: "#ea580c"}}>No Correlations Found</h4>
                    <p>The dataset may not have enough numeric variables or correlations are too weak to display.</p>
                    <details>
                      <summary>Debug Information</summary>
                      <div>
                        <p>Response received: {correlationData ? 'Yes' : 'No'}</p>
                        <p>Data structure: {JSON.stringify(Object.keys(correlationData?.data || {}))}</p>
                        <p>Correlations array length: {allCorrelations.length}</p>
                        <p>Insights array length: {fallbackInsights.length}</p>
                      </div>
                    </details>
                  </div>
                </div>
              ) : (
                <>
                  <details>
                    <summary>Debug: Raw Data Structure</summary>
                    <div className="stat-raw-data">
                      <pre className="stat-raw-row">
                        Correlations found: {allCorrelations.length}
                      </pre>
                      <pre className="stat-raw-row">
                        Insights found: {fallbackInsights.length}
                      </pre>
                      <pre className="stat-raw-row">
                        Matrix keys: {Object.keys(correlationMatrix).length}
                      </pre>
                      <pre className="stat-raw-head">
                        Raw Data:
                      </pre>
                      <pre className="stat-raw-col">
                        Success: {renderObjectAsDiv(correlationData?.success)}
                      </pre>
                      <pre className="stat-raw-col">
                        Analysis Type: {renderObjectAsDiv(correlationData?.analysis_type)}
                      </pre>
                      <pre className="stat-raw-col">
                        Dataset Id: {renderObjectAsDiv(correlationData?.dataset_id)}
                      </pre>
                      <pre className="stat-raw-col">
                        Timestamp: {correlationData?.timestamp ? new Date(correlationData.timestamp).toLocaleString() : ''}
                      </pre>
                      <pre className="stat-raw-col">
                        Data Insights: {renderObjectAsDiv(correlationData?.data?.insights)}
                      </pre>
                      <pre className="stat-raw-col">
                        Data Correlation: {renderObjectAsDiv(correlationData?.data?.correlations)}
                      </pre>
                      <pre className="stat-raw-col">
                        Strong Correlations: {renderObjectAsDiv(correlationData?.data?.strong_correlations)}
                      </pre>
                      <pre className="stat-raw-col">
                        Element Id Correlations: {renderObjectAsDiv(correlationData?.data?.correlation_matrix?.element_id)}
                      </pre>
                      <pre className="stat-raw-col">
                        Version Correlations: {renderObjectAsDiv(correlationData?.data?.correlation_matrix?.version)}
                      </pre>
                      <pre className="stat-raw-col">
                        Changeset Correlations: {renderObjectAsDiv(correlationData?.data?.correlation_matrix?.changeset)}
                      </pre>
                      <pre className="stat-raw-col">
                        Gauge Correlations: {renderObjectAsDiv(correlationData?.data?.correlation_matrix?.gauge)}
                      </pre>
                      <pre className="stat-raw-col">
                        Maxspeed Correlations: {renderObjectAsDiv(correlationData?.data?.correlation_matrix?.maxspeed)}
                      </pre>
                      <pre className="stat-raw-col">
                        Tracks Correlations: {renderObjectAsDiv(correlationData?.data?.correlation_matrix?.tracks)}
                      </pre>
                      <pre className="stat-raw-col">
                        Frequency Correlations: {renderObjectAsDiv(correlationData?.data?.correlation_matrix?.frequency)}
                      </pre>
                      <pre className="stat-raw-col">
                        Voltage Correlations: {renderObjectAsDiv(correlationData?.data?.correlation_matrix?.voltage)}
                      </pre>
                      <pre className="stat-raw-col">
                        Bridge Correlations: {renderObjectAsDiv(correlationData?.data?.correlation_matrix?.bridge)}
                      </pre>
                      <pre className="stat-raw-col">
                        Tunnel Correlations: {renderObjectAsDiv(correlationData?.data?.correlation_matrix?.tunnel)}
                      </pre>
                      <pre className="stat-raw-col">
                        Cutting Correlations: {renderObjectAsDiv(correlationData?.data?.correlation_matrix?.cutting)}
                      </pre>
                      <pre className="stat-raw-col">
                        Embankment Correlations: {renderObjectAsDiv(correlationData?.data?.correlation_matrix?.embankment)}
                      </pre>
                      <pre className="stat-raw-col">
                        Layer Correlations: {renderObjectAsDiv(correlationData?.data?.correlation_matrix?.layer)}
                      </pre>
                      <pre className="stat-raw-col">
                        Country Correlations: {renderObjectAsDiv(correlationData?.data?.correlation_matrix?.country)}
                      </pre>
                      <pre className="stat-raw-col">
                        State Correlations: {renderObjectAsDiv(correlationData?.data?.correlation_matrix?.state)}
                      </pre>
                      <pre className="stat-raw-col">
                        City Correlations: {renderObjectAsDiv(correlationData?.data?.correlation_matrix?.city)}
                      </pre>
                      <pre className="stat-raw-col">
                        Description Correlations: {renderObjectAsDiv(correlationData?.data?.correlation_matrix?.description)}
                      </pre>
                      <pre className="stat-raw-col">
                        Wikipedia Correlations: {renderObjectAsDiv(correlationData?.data?.correlation_matrix?.wikipedia)}
                      </pre>
                      <pre className="stat-raw-col">
                        Wikidata Correlations: {renderObjectAsDiv(correlationData?.data?.correlation_matrix?.wikidata)}
                      </pre>
                      <pre className="stat-raw-col">
                        latitude Correlations: {renderObjectAsDiv(correlationData?.data?.correlation_matrix?.lat)}
                      </pre>
                      <pre className="stat-raw-col">
                        longitude Correlations: {renderObjectAsDiv(correlationData?.data?.correlation_matrix?.lon)}
                      </pre>
                    </div>
                  </details>

                  {correlationInsights?.summary && (
                    <div className="corr-summary-box">
                      <h4 className="corr-summary-title">Summary</h4>
                      <p>{correlationInsights.summary}</p>
                    </div>
                  )}

                  {correlationInsights?.strong_positive_correlations?.length > 0 && (
                    <div className="corr-insight-box" style={{background: "#d1fae5", borderColor: "#059669"}}>
                      <h4 className="corr-insight-title" style={{color: "#059669"}}>Strong Positive Correlations</h4>
                      <div>
                        {correlationInsights.strong_positive_correlations.map((corr: any, index: number) => (
                          <div key={index} style={{marginBottom: "1rem"}}>
                            <div>
                              <span className="corr-top-label">{corr.variables[0]} ↔ {corr.variables[1]}</span>
                              <span style={{marginLeft: "1rem", color: "#059669"}}>r = {corr.correlation.toFixed(3)}</span>
                            </div>
                            <p>{corr.insight}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {correlationInsights?.strong_negative_correlations?.length > 0 && (
                    <div className="corr-insight-box" style={{background: "#fee2e2", borderColor: "#dc2626"}}>
                      <h4 className="corr-insight-title" style={{color: "#dc2626"}}>Strong Negative Correlations</h4>
                      <div>
                        {correlationInsights.strong_negative_correlations.map((corr: any, index: number) => (
                          <div key={index} style={{marginBottom: "1rem"}}>
                            <div>
                              <span className="corr-top-label">{corr.variables[0]} ↔ {corr.variables[1]}</span>
                              <span style={{marginLeft: "1rem", color: "#dc2626"}}>r = {corr.correlation.toFixed(3)}</span>
                            </div>
                            <p>{corr.insight}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {(generateInsights.length > 0 || fallbackInsights.length > 0) && (
                    <div className="corr-insight-box" style={{background: "#ede9fe", borderColor: "#7c3aed"}}>
                      <h4 className="corr-insight-title">Key Insights</h4>
                      <ul className="corr-insight-list">
                        {generateInsights.map((insight: string, index: number) => (
                          <li key={index} className="corr-insight-item">{insight}</li>
                        ))}
                        {fallbackInsights.map((insight: string, index: number) => (
                          <li key={`fallback-${index}`} className="corr-insight-item">{insight}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {allCorrelations.length > 0 && (
                    <div className="corr-insight-box" style={{background: "#e0e7ff", borderColor: "#6366f1"}}>
                      <h4 className="corr-insight-title">Correlation Strength Analysis</h4>
                      <div style={{display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "1rem"}}>
                        {processedCorrelations.strong.length > 0 && (
                          <div className="corr-insight-box" style={{background: "#fee2e2", borderColor: "#dc2626"}}>
                            <h5>Strong (|r| ≥ 0.7)</h5>
                            <ul className='ul-style'>
                              {processedCorrelations.strong.map((corr: CorrelationData, index: number) => (
                                <li key={index}>
                                  <span className="corr-top-label">{corr.var1} ↔ {corr.var2}</span>
                                  <span style={{marginLeft: "1rem", color: "#dc2626"}}>r = {corr.correlation.toFixed(3)}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                        {processedCorrelations.moderate.length > 0 && (
                          <div className="corr-insight-box" style={{background: "#fef9c3", borderColor: "#ca8a04"}}>
                            <h5>Moderate (0.3 ≤ |r| &lt; 0.7)</h5>
                            <ul className='ul-style'>
                              {processedCorrelations.moderate.slice(0, 5).map((corr: CorrelationData, index: number) => (
                                <li key={index}>
                                  <span className="corr-top-label">{corr.var1} ↔ {corr.var2}</span>
                                  <span style={{marginLeft: "1rem", color: "#ca8a04"}}>r = {corr.correlation.toFixed(3)}</span>
                                </li>
                              ))}
                              {processedCorrelations.moderate.length > 5 && (
                                <li style={{color: "#ca8a04", fontStyle: "italic"}}>
                                  +{processedCorrelations.moderate.length - 5} more...
                                </li>
                              )}
                            </ul>
                          </div>
                        )}
                        <div className="corr-insight-box" style={{background: "#dbeafe", borderColor: "#2563eb"}}>
                          <h5>Summary</h5>
                          <ul className='ul-style'>
                            <li>Positive: {allCorrelations.filter((c: CorrelationData) => c.correlation > 0).length}</li>
                            <li>Negative: {allCorrelations.filter((c: CorrelationData) => c.correlation < 0).length}</li>
                            <li>Avg |r|: {(allCorrelations.reduce((sum: number, c: CorrelationData) => sum + Math.abs(c.correlation), 0) / allCorrelations.length).toFixed(3)}</li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  )}

                  {correlationInsights?.multicollinearity_concerns?.length > 0 && (
                    <div className="corr-insight-box" style={{background: "#fef9c3", borderColor: "#ca8a04"}}>
                      <h4 className="corr-insight-title" style={{color: "#ca8a04"}}>Multicollinearity Concerns</h4>
                      <ul>
                        {correlationInsights.multicollinearity_concerns.map((concern: string, index: number) => (
                          <li key={index}>{concern}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {/* Recommendations Tab */}
          {activeTab === 'recommendations' && (
            <div className="corr-recommend-section">
              {generateRecommendations.length > 0 && (
                <div className="corr-recommend-box" style={{background: "#e0e7ff", borderColor: "#6366f1"}}>
                  <h4 className="corr-recommend-title">AI-Generated Recommendations</h4>
                  <ul className="corr-recommend-list">
                    {generateRecommendations.map((recommendation, index) => (
                      <li key={index} className="corr-recommend-item">{recommendation}</li>
                    ))}
                  </ul>
                </div>
              )}
              {allCorrelations.length > 0 && (
                <div className="corr-recommend-box" style={{background: "#d1fae5", borderColor: "#059669"}}>
                  <h4 className="corr-recommend-title">Specific Actions for Your Dataset</h4>
                  <ul className="corr-recommend-list">
                    {processedCorrelations.strong.length > 0 && (
                      <>
                        <li>Review the {processedCorrelations.strong.length} strong correlation(s) for feature redundancy</li>
                        <li>Consider using VIF (Variance Inflation Factor) analysis</li>
                        <li>Evaluate if both variables are needed for your analysis</li>
                      </>
                    )}
                    {processedCorrelations.moderate.length > 0 && (
                      <>
                        <li>Investigate the {processedCorrelations.moderate.length} moderate correlation(s) for business insights</li>
                        <li>These relationships might be meaningful for your domain</li>
                        <li>Consider feature engineering based on these relationships</li>
                      </>
                    )}
                    <li>Export correlation matrix for detailed analysis</li>
                    <li>Create scatter plots for top correlations</li>
                    <li>Document findings for stakeholder review</li>
                    <li>Consider correlation with target variable if doing predictive modeling</li>
                  </ul>
                </div>
              )}
              <div className="corr-recommend-box" style={{background: "#f1f5f9", borderColor: "#e5e7eb"}}>
                <h4 className="corr-recommend-title">Correlation Analysis Best Practices</h4>
                <div style={{display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem"}}>
                  <div>
                    <h5>Interpretation Guidelines</h5>
                    <ul className='ul-style'>
                      <li><strong>|r| ≥ 0.7:</strong> Strong correlation - investigate for multicollinearity</li>
                      <li><strong>0.3 ≤ |r| &lt; 0.7:</strong> Moderate correlation - potentially meaningful</li>
                      <li><strong>|r| &lt; 0.3:</strong> Weak correlation - limited linear relationship</li>
                    </ul>
                  </div>
                  <div>
                    <h5>Important Considerations</h5>
                    <ul className='ul-style'>
                      <li>Correlation ≠ Causation - always investigate further</li>
                      <li>Non-linear relationships may not show in Pearson correlation</li>
                      <li>Outliers can significantly affect correlation values</li>
                      <li>Consider domain knowledge when interpreting results</li>
                    </ul>
                  </div>
                </div>
              </div>
              {allCorrelations.length === 0 && (
                <div className="corr-no-data">
                  <p className="corr-title" style={{fontSize: '1.1rem'}}>No Correlation Data Available</p>
                  <p style={{fontSize: '0.95rem', marginTop: '0.5rem'}}>Ensure your dataset has at least 2 numeric columns to generate correlation insights</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CorrelationInsights;
