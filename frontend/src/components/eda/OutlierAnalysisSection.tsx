import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import aiService from '../../services/aiService';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface OutlierAnalysisSectionProps {
  datasetId: string;
}

const OutlierAnalysisSection: React.FC<OutlierAnalysisSectionProps> = ({ datasetId }) => {
  const [activeTab, setActiveTab] = useState('analysis');
  const [selectedMethod, setSelectedMethod] = useState('iqr');
  const [selectedColumn, setSelectedColumn] = useState('');
  const [treatmentMethod, setTreatmentMethod] = useState('cap_iqr');
  const [treatmentOptions, setTreatmentOptions] = useState({
    detection_method: 'iqr',
    z_threshold: 3.0,
    lower_percentile: 5,
    upper_percentile: 95
  });

  // Fetch outlier analysis
  const { data: analysisData, isLoading: analysisLoading, error: analysisError } = useQuery({
    queryKey: ['outlier-analysis', datasetId],
    queryFn: () => aiService.getOutlierAnalysis(datasetId),
    enabled: !!datasetId,
    retry: 2,
  });

  // Detection mutation
  const detectionMutation = useMutation({
    mutationFn: (options: any) => aiService.detectOutliers(datasetId, options),
  });

  // Treatment mutation
  const treatmentMutation = useMutation({
    mutationFn: (options: any) => aiService.treatOutliers(datasetId, treatmentMethod, options),
    onSuccess: (data) => {
      setActiveTab('results');
    },
  });

  // Preview mutation
  const previewMutation = useMutation({
    mutationFn: (options: any) => aiService.previewOutlierTreatment(datasetId, treatmentMethod, options),
  });

  // Download mutation
  const downloadMutation = useMutation({
    mutationFn: () => aiService.downloadTreatedDataset(datasetId),
    onSuccess: (blob) => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${datasetId}_outliers_treated.csv`;
      a.click();
      window.URL.revokeObjectURL(url);
    },
  });

  const analysis = analysisData?.data;
  const numericColumns = analysis ? Object.keys(analysis.column_analysis || {}) : [];

  const detectionMethods = [
    { value: 'z_score', label: '📊 Z-Score', description: 'Standard deviation based (threshold: 3)' },
    { value: 'modified_z_score', label: '📊 Modified Z-Score', description: 'Median-based robust method' },
    { value: 'iqr', label: '📦 IQR Method', description: 'Interquartile range (1.5 × IQR)' },
    { value: 'isolation_forest', label: '🌲 Isolation Forest', description: 'Machine learning approach' },
    { value: 'percentile', label: '📈 Percentile', description: 'Top/bottom percentiles (1%-99%)' },
  ];

  const treatmentMethods = [
    { value: 'remove', label: '🗑️ Remove Outliers', description: 'Delete rows with outliers' },
    { value: 'cap_iqr', label: '📊 Cap (IQR)', description: 'Cap to IQR bounds' },
    { value: 'cap_percentile', label: '📊 Cap (Percentile)', description: 'Cap to percentile bounds' },
    { value: 'cap_z_score', label: '📊 Cap (Z-Score)', description: 'Cap to Z-score bounds' },
    { value: 'winsorize', label: '🔄 Winsorize', description: 'Replace with percentile values' },
    { value: 'log_transform', label: '📈 Log Transform', description: 'Apply logarithmic transformation' },
    { value: 'sqrt_transform', label: '📈 Square Root Transform', description: 'Apply square root transformation' },
    { value: 'replace_median', label: '📊 Replace with Median', description: 'Replace outliers with median' },
    { value: 'replace_mean', label: '📊 Replace with Mean', description: 'Replace outliers with mean' },
  ];

  if (analysisLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mr-3"></div>
        <span>Analyzing outliers...</span>
      </div>
    );
  }

  if (analysisError) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <h4 className="font-medium text-red-900 mb-2">Error Loading Outlier Analysis</h4>
        <p className="text-sm text-red-800">Unable to analyze outliers. Please try again.</p>
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
        <div className="text-4xl mb-4">⚠️</div>
        <h4 className="font-medium text-yellow-900 mb-2">No Analysis Data</h4>
        <p className="text-sm text-yellow-800">Unable to load outlier analysis.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Tab Navigation */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6" aria-label="Tabs">
            {[
              { key: 'analysis', label: '📊 Analysis', icon: '📊' },
              { key: 'detection', label: '🔍 Detection', icon: '🔍' },
              { key: 'treatment', label: '🛠️ Treatment', icon: '🛠️' },
              { key: 'results', label: '📈 Results', icon: '📈' }
            ].map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.key
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        <div className="p-6">
          {/* Analysis Tab */}
          {activeTab === 'analysis' && (
            <div className="space-y-6">
              {/* Summary Statistics */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="bg-red-50 rounded-lg p-4">
                  <div className="text-2xl font-bold text-red-600">
                    {analysis.summary?.total_consensus_outliers || 0}
                  </div>
                  <div className="text-sm text-red-800">Total Outliers</div>
                </div>
                <div className="bg-yellow-50 rounded-lg p-4">
                  <div className="text-2xl font-bold text-yellow-600">
                    {analysis.summary?.columns_with_outliers || 0}
                  </div>
                  <div className="text-sm text-yellow-800">Columns Affected</div>
                </div>
                <div className="bg-blue-50 rounded-lg p-4">
                  <div className="text-2xl font-bold text-blue-600">
                    {analysis.summary?.consensus_outlier_percentage?.toFixed(1) || 0}%
                  </div>
                  <div className="text-sm text-blue-800">Outlier Percentage</div>
                </div>
              </div>

              {/* Column Analysis */}
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 mb-4">📋 Column-wise Outlier Analysis</h4>
                <div className="space-y-4">
                  {Object.entries(analysis.column_analysis || {}).map(([column, colData]: [string, any]) => (
                    <div key={column} className="bg-white rounded-lg p-4 border border-gray-200">
                      <div className="flex justify-between items-start mb-3">
                        <div>
                          <h5 className="font-medium text-gray-900">{column}</h5>
                          <p className="text-sm text-gray-600">
                            {colData.consensus_outliers?.length || 0} consensus outliers detected
                          </p>
                        </div>
                        <div className="text-right">
                          <div className="text-lg font-semibold text-red-600">
                            {((colData.consensus_outliers?.length || 0) / colData.total_values * 100).toFixed(1)}%
                          </div>
                          <div className="text-sm text-gray-600">of values</div>
                        </div>
                      </div>

                      {/* Detection Methods Results */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                        {Object.entries(colData.methods_results || {}).map(([method, result]: [string, any]) => (
                          <div key={method} className="bg-gray-50 rounded p-2">
                            <div className="text-xs font-medium text-gray-600 uppercase">{method.replace('_', ' ')}</div>
                            <div className="text-sm font-semibold text-gray-900">
                              {result.outlier_count} ({result.outlier_percentage?.toFixed(1)}%)
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Recommendations */}
                      {colData.recommended_treatment?.length > 0 && (
                        <div className="border-t border-gray-200 pt-3">
                          <h6 className="text-sm font-medium text-gray-700 mb-2">💡 Recommended Treatments:</h6>
                          <div className="space-y-1">
                            {colData.recommended_treatment.slice(0, 2).map((rec: any, idx: number) => (
                              <div key={idx} className="text-sm">
                                <span className="font-medium text-blue-600">{rec.method}</span>
                                <span className="text-gray-600 ml-2">- {rec.reason}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Global Recommendations */}
              {analysis.recommendations?.length > 0 && (
                <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4">
                  <h4 className="font-medium text-indigo-900 mb-3">🎯 Global Recommendations</h4>
                  <ul className="space-y-2">
                    {analysis.recommendations.map((rec: string, idx: number) => (
                      <li key={idx} className="flex items-start">
                        <span className="text-indigo-600 mr-2">•</span>
                        <span className="text-indigo-800 text-sm">{rec}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Detection Tab */}
          {activeTab === 'detection' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Detection Method</label>
                  <select
                    value={selectedMethod}
                    onChange={(e) => setSelectedMethod(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    {detectionMethods.map((method) => (
                      <option key={method.value} value={method.value}>
                        {method.label} - {method.description}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Column (Optional)</label>
                  <select
                    value={selectedColumn}
                    onChange={(e) => setSelectedColumn(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">All numeric columns</option>
                    {numericColumns.map((col) => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                </div>
              </div>

              <button
                onClick={() => detectionMutation.mutate({ method: selectedMethod, column: selectedColumn || undefined })}
                disabled={detectionMutation.isPending}
                className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
              >
                {detectionMutation.isPending ? 'Detecting...' : 'Detect Outliers'}
              </button>

              {/* Detection Results */}
              {detectionMutation.data && (
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-medium text-gray-900 mb-3">🔍 Detection Results</h4>
                  <div className="space-y-3">
                    {Object.entries(detectionMutation.data.detection_result.outliers_detected || {}).map(([col, count]: [string, any]) => (
                      <div key={col} className="flex justify-between items-center bg-white p-3 rounded border">
                        <span className="font-medium">{col}</span>
                        <span className="text-red-600 font-semibold">{count} outliers</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Treatment Tab */}
          {activeTab === 'treatment' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Treatment Method</label>
                  <select
                    value={treatmentMethod}
                    onChange={(e) => setTreatmentMethod(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    {treatmentMethods.map((method) => (
                      <option key={method.value} value={method.value}>
                        {method.label} - {method.description}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Column (Optional)</label>
                  <select
                    value={selectedColumn}
                    onChange={(e) => setSelectedColumn(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">All numeric columns</option>
                    {numericColumns.map((col) => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Treatment Options */}
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 mb-3">⚙️ Treatment Options</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Detection Method</label>
                    <select
                      value={treatmentOptions.detection_method}
                      onChange={(e) => setTreatmentOptions({...treatmentOptions, detection_method: e.target.value})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="iqr">IQR</option>
                      <option value="z_score">Z-Score</option>
                      <option value="modified_z_score">Modified Z-Score</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Lower Percentile</label>
                    <input
                      type="number"
                      min="0"
                      max="50"
                      step="0.1"
                      value={treatmentOptions.lower_percentile}
                      onChange={(e) => setTreatmentOptions({...treatmentOptions, lower_percentile: parseFloat(e.target.value)})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Upper Percentile</label>
                    <input
                      type="number"
                      min="50"
                      max="100"
                      step="0.1"
                      value={treatmentOptions.upper_percentile}
                      onChange={(e) => setTreatmentOptions({...treatmentOptions, upper_percentile: parseFloat(e.target.value)})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>
              </div>

              <div className="flex space-x-4">
                <button
                  onClick={() => previewMutation.mutate({...treatmentOptions, column: selectedColumn || undefined})}
                  disabled={previewMutation.isPending}
                  className="px-6 py-2 bg-yellow-600 text-white rounded-md hover:bg-yellow-700 disabled:opacity-50"
                >
                  {previewMutation.isPending ? 'Previewing...' : 'Preview Treatment'}
                </button>

                <button
                  onClick={() => treatmentMutation.mutate({...treatmentOptions, column: selectedColumn || undefined})}
                  disabled={treatmentMutation.isPending}
                  className="px-6 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
                >
                  {treatmentMutation.isPending ? 'Applying...' : 'Apply Treatment'}
                </button>
              </div>

              {/* Preview Results */}
              {previewMutation.data && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <h4 className="font-medium text-blue-900 mb-3">👁️ Treatment Preview</h4>
                  <div className="space-y-2">
                    <p><strong>Current Shape:</strong> {previewMutation.data.preview.current_shape[0]} rows × {previewMutation.data.preview.current_shape[1]} columns</p>
                    <p><strong>Estimated Final Shape:</strong> {previewMutation.data.preview.estimated_final_shape[0]} rows × {previewMutation.data.preview.estimated_final_shape[1]} columns</p>
                    {Object.entries(previewMutation.data.preview.estimated_changes || {}).map(([col, changes]: [string, any]) => (
                      <p key={col}><strong>{col}:</strong> {changes.estimated_impact}</p>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Results Tab */}
          {activeTab === 'results' && (
            <div className="space-y-6">
              {treatmentMutation.data ? (
                <div>
                  <div className="bg-green-50 border border-green-200 rounded-lg p-6 mb-6">
                    <div className="flex items-center mb-4">
                      <div className="text-3xl mr-3">✅</div>
                      <div>
                        <h4 className="font-medium text-green-900">Treatment Applied Successfully!</h4>
                        <p className="text-sm text-green-800">Your dataset has been processed and is ready for download.</p>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                      <div className="text-center">
                        <div className="text-lg font-semibold text-green-700">
                          {treatmentMutation.data.treatment_info.original_shape[0]}
                        </div>
                        <div className="text-sm text-green-600">Original Rows</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-semibold text-green-700">
                          {treatmentMutation.data.treatment_info.final_shape[0]}
                        </div>
                        <div className="text-sm text-green-600">Final Rows</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-semibold text-green-700">
                          {treatmentMutation.data.treatment_info.outliers_removed || treatmentMutation.data.treatment_info.outliers_treated || 0}
                        </div>
                        <div className="text-sm text-green-600">Outliers Processed</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-semibold text-green-700">
                          {treatmentMutation.data.treatment_info.method}
                        </div>
                        <div className="text-sm text-green-600">Method Used</div>
                      </div>
                    </div>

                    <button
                      onClick={() => {
                        const treatedId = treatmentMutation.data.treated_dataset_id;
                        window.open(`${API_BASE_URL}/api/v1/ai-insights/${treatedId}/download/csv`, '_blank');
                      }}
                      className="w-full px-6 py-3 bg-green-600 text-white rounded-md hover:bg-green-700 font-medium flex items-center justify-center space-x-2"
                    >
                      <span>📥</span>
                      <span>Download Treated Dataset (No Outliers)</span>
                    </button>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <div className="text-4xl mb-4">📊</div>
                  <p>Apply a treatment method to see results here.</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default OutlierAnalysisSection;
