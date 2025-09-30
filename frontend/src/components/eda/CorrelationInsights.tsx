import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import Plot from 'react-plotly.js';
import aiService from '../../services/aiService';

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
  const visualizationData = propCorrelationData;

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
                  
                  {/* Quick Stats */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-blue-50 rounded-lg p-4">
                      <div className="text-2xl font-bold text-blue-600">
                        {correlationInsights?.strong_positive_correlations?.length || 0}
                      </div>
                      <div className="text-sm text-blue-800">Strong Positive Correlations</div>
                    </div>
                    <div className="bg-red-50 rounded-lg p-4">
                      <div className="text-2xl font-bold text-red-600">
                        {correlationInsights?.strong_negative_correlations?.length || 0}
                      </div>
                      <div className="text-sm text-red-800">Strong Negative Correlations</div>
                    </div>
                    <div className="bg-yellow-50 rounded-lg p-4">
                      <div className="text-2xl font-bold text-yellow-600">
                        {correlationInsights?.multicollinearity_concerns?.length || 0}
                      </div>
                      <div className="text-sm text-yellow-800">Multicollinearity Concerns</div>
                    </div>
                  </div>
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
              ) : (
                <>
                  {/* Debug Data Structure */}
                  <details className="mb-4">
                    <summary className="cursor-pointer text-sm text-gray-600">Debug: Raw Data Structure</summary>
                    <pre className="text-xs mt-2 bg-gray-100 p-2 rounded overflow-auto max-h-40">
                      {JSON.stringify(enhancedData, null, 2)}
                    </pre>
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
                        {correlationInsights.strong_positive_correlations.map((corr, index) => (
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
                        {correlationInsights.strong_negative_correlations.map((corr, index) => (
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

                  {/* Key Insights */}
                  {(correlationInsights?.key_insights?.length > 0 || fallbackInsights.length > 0) && (
                    <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                      <h4 className="font-medium text-purple-900 mb-3">🔍 Key Insights</h4>
                      <ul className="space-y-2">
                        {(correlationInsights?.key_insights || fallbackInsights).map((insight, index) => (
                          <li key={index} className="flex items-start">
                            <span className="text-purple-600 mr-2">•</span>
                            <span className="text-purple-800">{insight}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Multicollinearity Concerns */}
                  {correlationInsights?.multicollinearity_concerns?.length > 0 && (
                    <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                      <h4 className="font-medium text-yellow-900 mb-3">⚠️ Multicollinearity Concerns</h4>
                      <ul className="space-y-2">
                        {correlationInsights.multicollinearity_concerns.map((concern, index) => (
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

          {/* Recommendations Tab */}
          {activeTab === 'recommendations' && (
            <div className="space-y-6">
              {correlationInsights?.recommendations?.length > 0 ? (
                <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4">
                  <h4 className="font-medium text-indigo-900 mb-3">💡 AI Recommendations</h4>
                  <div className="space-y-3">
                    {correlationInsights.recommendations.map((recommendation, index) => (
                      <div key={index} className="bg-white rounded-lg p-3 border border-indigo-100">
                        <div className="flex items-start">
                          <span className="bg-indigo-100 text-indigo-800 px-2 py-1 rounded text-xs font-medium mr-3 mt-0.5">
                            {index + 1}
                          </span>
                          <span className="text-indigo-800">{recommendation}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 text-gray-500">
                  <div className="text-4xl mb-4">💡</div>
                  <p>No specific recommendations available</p>
                  <p className="text-sm">AI insights will appear here once correlation analysis is complete</p>
                </div>
              )}

              {/* General Best Practices */}
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 mb-3">📚 General Best Practices</h4>
                <ul className="space-y-2 text-sm text-gray-700">
                  <li className="flex items-start">
                    <span className="text-gray-500 mr-2">•</span>
                    <span>Correlations above 0.7 or below -0.7 are considered strong</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gray-500 mr-2">•</span>
                    <span>High correlations between features may indicate multicollinearity</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gray-500 mr-2">•</span>
                    <span>Consider removing highly correlated features to improve model performance</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-gray-500 mr-2">•</span>
                    <span>Correlation does not imply causation - investigate relationships further</span>
                  </li>
                </ul>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CorrelationInsights;
