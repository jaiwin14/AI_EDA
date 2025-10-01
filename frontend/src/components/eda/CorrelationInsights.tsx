import React, { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import Plot from 'react-plotly.js';
import aiService from '../../services/aiService';

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

  // Fetch enhanced correlation analysis
  const { data: enhancedData, isLoading: aiLoading, error: aiError } = useQuery({
    queryKey: ['enhanced-correlation-analysis', datasetId],
    queryFn: () => aiService.getEnhancedCorrelationAnalysis(datasetId),
    enabled: !!datasetId,
    retry: 2,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  const correlationInsights = enhancedData?.data?.correlation_insights;
  const fallbackInsights = enhancedData?.data?.insights || [];
  const allCorrelations: CorrelationData[] = enhancedData?.data?.correlations || [];
  const correlationMatrix = enhancedData?.data?.correlation_matrix || {};
  const visualizationData = propCorrelationData;

  // Debug logging
  React.useEffect(() => {
    if (enhancedData) {
      console.log('🔍 Correlation Debug - Enhanced Data:', enhancedData);
      console.log('📊 Correlations found:', allCorrelations.length);
      console.log('💡 Insights found:', fallbackInsights.length);
      console.log('🔗 Matrix keys:', Object.keys(correlationMatrix).length);
    }
  }, [enhancedData, allCorrelations, fallbackInsights, correlationMatrix]);

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
    insights.push(`📊 Analyzed ${totalVars} numeric variables with ${allCorrelations.length} correlation pairs`);
    
    // Strong correlations
    if (processedCorrelations.strong.length > 0) {
      insights.push(`🔴 Found ${processedCorrelations.strong.length} strong correlations (|r| ≥ 0.7)`);
      const strongestCorr = processedCorrelations.strong[0];
      const direction = strongestCorr.correlation > 0 ? 'positive' : 'negative';
      insights.push(`📈 Strongest ${direction} correlation: ${strongestCorr.var1} ↔ ${strongestCorr.var2} (r = ${strongestCorr.correlation.toFixed(3)})`);
    }
    
    // Moderate correlations
    if (processedCorrelations.moderate.length > 0) {
      insights.push(`🟡 Found ${processedCorrelations.moderate.length} moderate correlations (0.3 <= |r| < 0.7)`);
    }
    
    // Multicollinearity warning
    if (processedCorrelations.strong.length > 0) {
      insights.push(`⚠️ Strong correlations detected - consider checking for multicollinearity in modeling`);
    }
    
    // Data quality insights
    const positiveCorrs = allCorrelations.filter((c: CorrelationData) => c.correlation > 0).length;
    const negativeCorrs = allCorrelations.filter((c: CorrelationData) => c.correlation < 0).length;
    insights.push(`📊 Distribution: ${positiveCorrs} positive, ${negativeCorrs} negative correlations`);
    
    return insights;
  }, [allCorrelations, correlationMatrix, processedCorrelations]);

  // Generate recommendations
  const generateRecommendations = useMemo(() => {
    if (!allCorrelations.length) return [];
    
    const recommendations = [];
    
    // Strong correlation recommendations
    if (processedCorrelations.strong.length > 0) {
      recommendations.push('🔍 Investigate strong correlations for potential feature redundancy');
      recommendations.push('📉 Consider removing highly correlated features to reduce multicollinearity');
      recommendations.push('🧪 Use techniques like PCA or feature selection to handle correlated features');
    }
    
    // Moderate correlation recommendations
    if (processedCorrelations.moderate.length > 0) {
      recommendations.push('📊 Moderate correlations may indicate meaningful relationships worth exploring');
      recommendations.push('🔬 Consider domain knowledge to interpret moderate correlations');
    }
    
    // General recommendations
    recommendations.push('📈 Remember: correlation does not imply causation');
    recommendations.push('🎯 Focus on correlations relevant to your target variable for predictive modeling');
    recommendations.push('📋 Document significant correlations for stakeholder communication');
    
    return recommendations;
  }, [allCorrelations, processedCorrelations]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-xl font-semibold text-gray-900 mb-2">
          🔗 Correlation Analysis
        </h3>
        <p className="text-gray-600">
          Explore relationships between numeric variables with AI-powered insights
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6" aria-label="Tabs">
            {[
              { key: 'heatmap', label: '📊 Correlation Matrix', icon: '📊' },
              { key: 'insights', label: '🧠 AI Insights', icon: '🧠' },
              { key: 'recommendations', label: '💡 Recommendations', icon: '💡' }
            ].map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key as any)}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.key
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        <div className="p-6">
          {/* Correlation Heatmap Tab */}
          {activeTab === 'heatmap' && (
            <div>
              {visualizationData?.figure ? (
                <div className="space-y-4">
                  <div className="bg-gray-50 rounded-lg p-4">
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
                  
                  {/* Enhanced Quick Stats */}
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div className="bg-red-50 rounded-lg p-4">
                      <div className="text-2xl font-bold text-red-600">
                        {processedCorrelations.strong.length}
                      </div>
                      <div className="text-sm text-red-800">Strong Correlations</div>
                      <div className="text-xs text-red-600 mt-1">|r| ≥ 0.7</div>
                    </div>
                    <div className="bg-yellow-50 rounded-lg p-4">
                      <div className="text-2xl font-bold text-yellow-600">
                        {processedCorrelations.moderate.length}
                      </div>
                      <div className="text-sm text-yellow-800">Moderate Correlations</div>
                      <div className="text-xs text-yellow-600 mt-1">0.3 &le; |r| &lt; 0.7</div>
                    </div>
                    <div className="bg-green-50 rounded-lg p-4">
                      <div className="text-2xl font-bold text-green-600">
                        {processedCorrelations.weak.length}
                      </div>
                      <div className="text-sm text-green-800">Weak Correlations</div>
                      <div className="text-xs text-green-600 mt-1">|r| &lt; 0.3</div>
                    </div>
                    <div className="bg-blue-50 rounded-lg p-4">
                      <div className="text-2xl font-bold text-blue-600">
                        {Object.keys(correlationMatrix).length}
                      </div>
                      <div className="text-sm text-blue-800">Variables Analyzed</div>
                      <div className="text-xs text-blue-600 mt-1">Numeric only</div>
                    </div>
                  </div>
                  
                  {/* Top Correlations List */}
                  {allCorrelations.length > 0 && (
                    <div className="bg-white rounded-lg border border-gray-200 p-4">
                      <h4 className="font-medium text-gray-900 mb-3">🔝 Top Correlations</h4>
                      <div className="space-y-2 max-h-64 overflow-y-auto">
                        {allCorrelations.slice(0, 10).map((corr: CorrelationData, index: number) => {
                          const strength = Math.abs(corr.correlation) >= 0.7 ? 'strong' : 
                                         Math.abs(corr.correlation) >= 0.3 ? 'moderate' : 'weak';
                          const strengthColor = strength === 'strong' ? 'text-red-600 bg-red-50' :
                                               strength === 'moderate' ? 'text-yellow-600 bg-yellow-50' :
                                               'text-green-600 bg-green-50';
                          const direction = corr.correlation > 0 ? '📈' : '📉';
                          
                          return (
                            <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                              <div className="flex items-center space-x-2">
                                <span className="text-lg">{direction}</span>
                                <span className="font-medium text-sm">
                                  {corr.var1} ↔ {corr.var2}
                                </span>
                              </div>
                              <div className="flex items-center space-x-2">
                                <span className={`px-2 py-1 rounded text-xs font-medium ${strengthColor}`}>
                                  {strength}
                                </span>
                                <span className="font-mono text-sm font-bold">
                                  {corr.correlation.toFixed(3)}
                                </span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-12 text-gray-500">
                  <div className="text-4xl mb-4">📊</div>
                  <p>No correlation heatmap available</p>
                  <p className="text-sm">Ensure your dataset has at least 2 numeric columns</p>
                </div>
              )}
            </div>
          )}

          {/* AI Insights Tab */}
          {activeTab === 'insights' && (
            <div className="space-y-6">
              {aiLoading ? (
                <div className="text-center py-8">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
                  <p className="text-gray-600">Generating AI insights...</p>
                </div>
              ) : aiError ? (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <div className="text-red-800">
                    <h4 className="font-medium mb-2">Error Loading AI Insights</h4>
                    <p className="text-sm">Unable to generate correlation insights. Please try again.</p>
                    <details className="mt-2">
                      <summary className="cursor-pointer text-xs">Debug Info</summary>
                      <pre className="text-xs mt-2 bg-red-100 p-2 rounded overflow-auto">
                        {JSON.stringify(aiError, null, 2)}
                      </pre>
                    </details>
                  </div>
                </div>
              ) : !enhancedData ? (
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                  <div className="text-yellow-800">
                    <h4 className="font-medium mb-2">No Data Available</h4>
                    <p className="text-sm">No correlation data received from the server.</p>
                  </div>
                </div>
              ) : allCorrelations.length === 0 ? (
                <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
                  <div className="text-orange-800">
                    <h4 className="font-medium mb-2">🔍 No Correlations Found</h4>
                    <p className="text-sm">The dataset may not have enough numeric variables or correlations are too weak to display.</p>
                    <details className="mt-3">
                      <summary className="cursor-pointer text-sm font-medium">Debug Information</summary>
                      <div className="mt-2 text-xs bg-orange-100 p-2 rounded">
                        <p>Response received: {enhancedData ? 'Yes' : 'No'}</p>
                        <p>Data structure: {JSON.stringify(Object.keys(enhancedData?.data || {}))}</p>
                        <p>Correlations array length: {allCorrelations.length}</p>
                        <p>Insights array length: {fallbackInsights.length}</p>
                      </div>
                    </details>
                  </div>
                </div>
              ) : (
                <>
                  {/* Debug Data Structure */}
                  <details className="mb-4">
                    <summary className="cursor-pointer text-sm text-gray-600">Debug: Raw Data Structure</summary>
                    <pre className="text-xs mt-2 bg-gray-100 p-2 rounded overflow-auto max-h-40">
                      {JSON.stringify(enhancedData, null, 2)}
                    </pre>
                    <div className="mt-2 text-xs text-gray-600">
                      <p>Correlations found: {allCorrelations.length}</p>
                      <p>Insights found: {fallbackInsights.length}</p>
                      <p>Matrix keys: {Object.keys(correlationMatrix).length}</p>
                    </div>
                  </details>

                  {/* Summary */}
                  {correlationInsights?.summary && (
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                      <h4 className="font-medium text-blue-900 mb-2">📋 Summary</h4>
                      <p className="text-blue-800">{correlationInsights.summary}</p>
                    </div>
                  )}

                  {/* Strong Positive Correlations */}
                  {correlationInsights?.strong_positive_correlations?.length > 0 && (
                    <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                      <h4 className="font-medium text-green-900 mb-3">📈 Strong Positive Correlations</h4>
                      <div className="space-y-3">
                        {correlationInsights.strong_positive_correlations.map((corr: any, index: number) => (
                          <div key={index} className="bg-white rounded-lg p-3 border border-green-100">
                            <div className="flex justify-between items-start mb-2">
                              <span className="font-medium text-gray-900">
                                {corr.variables[0]} ↔ {corr.variables[1]}
                              </span>
                              <span className="bg-green-100 text-green-800 px-2 py-1 rounded text-sm font-medium">
                                r = {corr.correlation.toFixed(3)}
                              </span>
                            </div>
                            <p className="text-gray-700 text-sm">{corr.insight}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Strong Negative Correlations */}
                  {correlationInsights?.strong_negative_correlations?.length > 0 && (
                    <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                      <h4 className="font-medium text-red-900 mb-3">📉 Strong Negative Correlations</h4>
                      <div className="space-y-3">
                        {correlationInsights.strong_negative_correlations.map((corr: any, index: number) => (
                          <div key={index} className="bg-white rounded-lg p-3 border border-red-100">
                            <div className="flex justify-between items-start mb-2">
                              <span className="font-medium text-gray-900">
                                {corr.variables[0]} ↔ {corr.variables[1]}
                              </span>
                              <span className="bg-red-100 text-red-800 px-2 py-1 rounded text-sm font-medium">
                                r = {corr.correlation.toFixed(3)}
                              </span>
                            </div>
                            <p className="text-gray-700 text-sm">{corr.insight}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Enhanced Key Insights */}
                  {(generateInsights.length > 0 || fallbackInsights.length > 0) && (
                    <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                      <h4 className="font-medium text-purple-900 mb-3">🔍 Key Insights</h4>
                      <div className="space-y-3">
                        {generateInsights.map((insight: string, index: number) => (
                          <div key={index} className="flex items-start bg-white rounded-lg p-3 border border-purple-100">
                            <span className="text-purple-600 mr-3 mt-0.5">•</span>
                            <span className="text-purple-800 text-sm leading-relaxed">{insight}</span>
                          </div>
                        ))}
                        {fallbackInsights.map((insight: string, index: number) => (
                          <div key={`fallback-${index}`} className="flex items-start bg-white rounded-lg p-3 border border-purple-100">
                            <span className="text-purple-600 mr-3 mt-0.5">•</span>
                            <span className="text-purple-800 text-sm leading-relaxed">{insight}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {/* Correlation Strength Breakdown */}
                  {allCorrelations.length > 0 && (
                    <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4">
                      <h4 className="font-medium text-indigo-900 mb-3">📊 Correlation Strength Analysis</h4>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {/* Strong Correlations */}
                        {processedCorrelations.strong.length > 0 && (
                          <div className="bg-white rounded-lg p-3 border border-red-200">
                            <h5 className="font-medium text-red-900 mb-2 flex items-center">
                              🔴 Strong (|r| ≥ 0.7)
                            </h5>
                            <div className="space-y-1 max-h-32 overflow-y-auto">
                              {processedCorrelations.strong.map((corr: CorrelationData, index: number) => (
                                <div key={index} className="text-xs">
                                  <div className="font-medium text-gray-900">
                                    {corr.var1} ↔ {corr.var2}
                                  </div>
                                  <div className="text-red-600 font-mono">
                                    r = {corr.correlation.toFixed(3)}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                        
                        {/* Moderate Correlations */}
                        {processedCorrelations.moderate.length > 0 && (
                          <div className="bg-white rounded-lg p-3 border border-yellow-200">
                            <h5 className="font-medium text-yellow-900 mb-2 flex items-center">
                              🟡 Moderate (0.3 &le; |r| &lt; 0.7)
                            </h5>
                            <div className="space-y-1 max-h-32 overflow-y-auto">
                              {processedCorrelations.moderate.slice(0, 5).map((corr: CorrelationData, index: number) => (
                                <div key={index} className="text-xs">
                                  <div className="font-medium text-gray-900">
                                    {corr.var1} ↔ {corr.var2}
                                  </div>
                                  <div className="text-yellow-600 font-mono">
                                    r = {corr.correlation.toFixed(3)}
                                  </div>
                                </div>
                              ))}
                              {processedCorrelations.moderate.length > 5 && (
                                <div className="text-xs text-yellow-600 italic">
                                  +{processedCorrelations.moderate.length - 5} more...
                                </div>
                              )}
                            </div>
                          </div>
                        )}
                        
                        {/* Summary Stats */}
                        <div className="bg-white rounded-lg p-3 border border-indigo-200">
                          <h5 className="font-medium text-indigo-900 mb-2">📈 Summary</h5>
                          <div className="space-y-1 text-xs">
                            <div className="flex justify-between">
                              <span>Positive:</span>
                              <span className="font-medium">
                                {allCorrelations.filter((c: CorrelationData) => c.correlation > 0).length}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span>Negative:</span>
                              <span className="font-medium">
                                {allCorrelations.filter((c: CorrelationData) => c.correlation < 0).length}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span>Avg |r|:</span>
                              <span className="font-medium font-mono">
                                {(allCorrelations.reduce((sum: number, c: CorrelationData) => sum + Math.abs(c.correlation), 0) / allCorrelations.length).toFixed(3)}
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Multicollinearity Concerns */}
                  {correlationInsights?.multicollinearity_concerns?.length > 0 && (
                    <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                      <h4 className="font-medium text-yellow-900 mb-3">⚠️ Multicollinearity Concerns</h4>
                      <ul className="space-y-2">
                        {correlationInsights.multicollinearity_concerns.map((concern: string, index: number) => (
                          <li key={index} className="flex items-start">
                            <span className="text-yellow-600 mr-2">•</span>
                            <span className="text-yellow-800">{concern}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {/* Enhanced Recommendations Tab */}
          {activeTab === 'recommendations' && (
            <div className="space-y-6">
              {/* AI-Generated Recommendations */}
              {generateRecommendations.length > 0 && (
                <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4">
                  <h4 className="font-medium text-indigo-900 mb-3">🤖 AI-Generated Recommendations</h4>
                  <div className="space-y-3">
                    {generateRecommendations.map((recommendation, index) => (
                      <div key={index} className="bg-white rounded-lg p-4 border border-indigo-100 shadow-sm">
                        <div className="flex items-start">
                          <span className="bg-indigo-100 text-indigo-800 px-2 py-1 rounded-full text-xs font-medium mr-3 mt-0.5 min-w-[24px] text-center">
                            {index + 1}
                          </span>
                          <span className="text-indigo-800 text-sm leading-relaxed">{recommendation}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Specific Actions Based on Data */}
              {allCorrelations.length > 0 && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <h4 className="font-medium text-green-900 mb-3">🎯 Specific Actions for Your Dataset</h4>
                  <div className="space-y-3">
                    {processedCorrelations.strong.length > 0 && (
                      <div className="bg-white rounded-lg p-3 border border-green-100">
                        <h5 className="font-medium text-green-800 mb-2">🔴 Strong Correlations Detected</h5>
                        <ul className="text-sm text-green-700 space-y-1">
                          <li>• Review the {processedCorrelations.strong.length} strong correlation(s) for feature redundancy</li>
                          <li>• Consider using VIF (Variance Inflation Factor) analysis</li>
                          <li>• Evaluate if both variables are needed for your analysis</li>
                        </ul>
                      </div>
                    )}
                    
                    {processedCorrelations.moderate.length > 0 && (
                      <div className="bg-white rounded-lg p-3 border border-green-100">
                        <h5 className="font-medium text-green-800 mb-2">🟡 Moderate Correlations Found</h5>
                        <ul className="text-sm text-green-700 space-y-1">
                          <li>• Investigate the {processedCorrelations.moderate.length} moderate correlation(s) for business insights</li>
                          <li>• These relationships might be meaningful for your domain</li>
                          <li>• Consider feature engineering based on these relationships</li>
                        </ul>
                      </div>
                    )}
                    
                    <div className="bg-white rounded-lg p-3 border border-green-100">
                      <h5 className="font-medium text-green-800 mb-2">📊 Next Steps</h5>
                      <ul className="text-sm text-green-700 space-y-1">
                        <li>• Export correlation matrix for detailed analysis</li>
                        <li>• Create scatter plots for top correlations</li>
                        <li>• Document findings for stakeholder review</li>
                        <li>• Consider correlation with target variable if doing predictive modeling</li>
                      </ul>
                    </div>
                  </div>
                </div>
              )}

              {/* Enhanced Best Practices */}
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 mb-3">📚 Correlation Analysis Best Practices</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-3">
                    <h5 className="font-medium text-gray-800">🎯 Interpretation Guidelines</h5>
                    <ul className="space-y-2 text-sm text-gray-700">
                      <li className="flex items-start">
                        <span className="text-red-500 mr-2 font-bold">•</span>
                        <span><strong>|r| ≥ 0.7:</strong> Strong correlation - investigate for multicollinearity</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-yellow-500 mr-2 font-bold">•</span>
                        <span><strong>0.3 &le; |r| &lt; 0.7:</strong> Moderate correlation - potentially meaningful</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-green-500 mr-2 font-bold">•</span>
                        <span><strong>|r| &lt; 0.3:</strong> Weak correlation - limited linear relationship</span>
                      </li>
                    </ul>
                  </div>
                  
                  <div className="space-y-3">
                    <h5 className="font-medium text-gray-800">⚠️ Important Considerations</h5>
                    <ul className="space-y-2 text-sm text-gray-700">
                      <li className="flex items-start">
                        <span className="text-gray-500 mr-2">•</span>
                        <span>Correlation &ne; Causation - always investigate further</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-gray-500 mr-2">•</span>
                        <span>Non-linear relationships may not show in Pearson correlation</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-gray-500 mr-2">•</span>
                        <span>Outliers can significantly affect correlation values</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-gray-500 mr-2">•</span>
                        <span>Consider domain knowledge when interpreting results</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </div>
              
              {/* No Data State */}
              {allCorrelations.length === 0 && (
                <div className="text-center py-12 text-gray-500">
                  <div className="text-4xl mb-4">💡</div>
                  <p className="text-lg font-medium mb-2">No Correlation Data Available</p>
                  <p className="text-sm">Ensure your dataset has at least 2 numeric columns to generate correlation insights</p>
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
