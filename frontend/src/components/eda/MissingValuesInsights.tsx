import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import aiService from '../../services/aiService';

interface MissingValuesInsightsProps {
  datasetId: string;
}

interface ColumnAnalysis {
  missing_count: number;
  missing_percentage: number;
  data_type: string;
  unique_values: number;
  pattern_type: string;
  reasons: string[];
  recommended_methods: Array<{
    method: string;
    priority: number;
    reason: string;
    pros: string[];
    cons: string[];
  }>;
}

interface TreatmentOptions {
  method: string;
  column?: string;
  n_neighbors?: number;
  constant_value?: any;
  threshold?: number;
}

const MissingValuesInsights: React.FC<MissingValuesInsightsProps> = ({ datasetId }) => {
  const [activeTab, setActiveTab] = useState<'analysis' | 'treatment' | 'results'>('analysis');
  const [selectedMethod, setSelectedMethod] = useState<string>('');
  const [selectedColumn, setSelectedColumn] = useState<string>('');
  const [treatmentOptions, setTreatmentOptions] = useState<TreatmentOptions>({
    method: '',
    n_neighbors: 5,
    constant_value: 0,
    threshold: 0.5
  });
  const [showPreview, setShowPreview] = useState(false);

  const queryClient = useQueryClient();

  // Fetch missing values analysis
  const { data: analysisData, isLoading: analysisLoading, error: analysisError } = useQuery({
    queryKey: ['missing-values-analysis', datasetId],
    queryFn: () => aiService.getMissingValuesInsights(datasetId),
    enabled: !!datasetId,
    retry: 2,
    staleTime: 5 * 60 * 1000,
  });

  // Preview treatment mutation
  const previewMutation = useMutation({
    mutationFn: (options: TreatmentOptions) => 
      aiService.previewMissingValuesTreatment(datasetId, options.method, options),
    onSuccess: () => setShowPreview(true),
  });

  // Apply treatment mutation
  const treatmentMutation = useMutation({
    mutationFn: (options: TreatmentOptions) => 
      aiService.treatMissingValues(datasetId, options.method, options),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['missing-values-analysis'] });
      setActiveTab('results');
    },
  });

  // Download original dataset mutation
  const downloadOriginalMutation = useMutation({
    mutationFn: () => aiService.downloadOriginalDataset(datasetId),
    onSuccess: (blob) => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${datasetId}_original.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    },
  });

  // Download treated dataset mutation
  const downloadTreatedMutation = useMutation({
    mutationFn: () => aiService.downloadTreatedDataset(datasetId),
    onSuccess: (blob) => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${datasetId}_cleaned.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    },
  });

  const analysis = analysisData?.data;
  const columnAnalysis: Record<string, ColumnAnalysis> = analysis?.column_analysis || {};
  const missingColumns = Object.keys(columnAnalysis);

  // Debug logging
  React.useEffect(() => {
    if (analysisData) {
      console.log('🔍 Missing Values Debug - Full Response:', analysisData);
      console.log('📊 Analysis Data:', analysis);
      console.log('📋 Column Analysis:', columnAnalysis);
      console.log('🔗 Missing Columns:', missingColumns);
    }
  }, [analysisData, analysis, columnAnalysis, missingColumns]);

  // Treatment methods mapping
  const treatmentMethods = [
    { value: 'drop_rows', label: '🗑️ Drop Rows', description: 'Remove rows with missing values' },
    { value: 'drop_columns', label: '🗑️ Drop Columns', description: 'Remove columns with high missing percentage' },
    { value: 'mean_imputation', label: '📊 Mean Imputation', description: 'Fill with column mean (numeric only)' },
    { value: 'median_imputation', label: '📊 Median Imputation', description: 'Fill with column median (numeric only)' },
    { value: 'mode_imputation', label: '📊 Mode Imputation', description: 'Fill with most frequent value' },
    { value: 'forward_fill', label: '⏭️ Forward Fill', description: 'Fill with previous value' },
    { value: 'backward_fill', label: '⏮️ Backward Fill', description: 'Fill with next value' },
    { value: 'knn_imputation', label: '🔍 KNN Imputation', description: 'Fill using K-nearest neighbors' },
    { value: 'iterative_imputation', label: '🔄 Iterative Imputation', description: 'Advanced multivariate imputation' },
    { value: 'constant_imputation', label: '🔢 Constant Fill', description: 'Fill with constant value' },
  ];

  const handlePreview = () => {
    if (selectedMethod) {
      const { method: _, ...options } = treatmentOptions;
      previewMutation.mutate({
        method: selectedMethod,
        column: selectedColumn || undefined,
        ...options
      });
    }
  };

  const handleTreatment = () => {
    if (selectedMethod) {
      const { method: _, ...options } = treatmentOptions;
      treatmentMutation.mutate({
        method: selectedMethod,
        column: selectedColumn || undefined,
        ...options
      });
    }
  };

  const getPatternColor = (pattern: string) => {
    switch (pattern) {
      case 'Missing Completely At Random': return 'text-green-600 bg-green-50';
      case 'Missing At Random': return 'text-yellow-600 bg-yellow-50';
      case 'Missing Not At Random': return 'text-red-600 bg-red-50';
      case 'Structural Missing': return 'text-purple-600 bg-purple-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  if (analysisLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-3 text-gray-600">Analyzing missing values...</span>
      </div>
    );
  }

  if (analysisError) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <h4 className="font-medium text-red-900 mb-2">Error Loading Analysis</h4>
        <p className="text-sm text-red-800">Unable to analyze missing values. Please try again.</p>
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
        <div className="text-4xl mb-4">⚠️</div>
        <h4 className="font-medium text-yellow-900 mb-2">No Analysis Data</h4>
        <p className="text-sm text-yellow-800">Unable to load missing values analysis.</p>
        <details className="mt-4 text-left">
          <summary className="cursor-pointer text-sm font-medium">Debug Information</summary>
          <div className="mt-2 text-xs bg-yellow-100 p-2 rounded">
            <p>Response: {analysisData ? 'Received' : 'None'}</p>
            <p>Data structure: {JSON.stringify(analysisData, null, 2)}</p>
          </div>
        </details>
      </div>
    );
  }

  if (missingColumns.length === 0) {
    return (
      <div className="space-y-6">
        {/* Header with download button */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex justify-between items-start">
            <div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                🔍 Missing Values Analysis & Treatment
              </h3>
              <p className="text-gray-600">
                Your dataset is complete with no missing values!
              </p>
            </div>
            <div className="flex space-x-3">
              <button
                onClick={() => downloadOriginalMutation.mutate()}
                disabled={downloadOriginalMutation.isPending}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
              >
                <span>📥</span>
                <span>{downloadOriginalMutation.isPending ? 'Downloading...' : 'Download Original CSV'}</span>
              </button>
            </div>
          </div>
        </div>
        
        <div className="bg-green-50 border border-green-200 rounded-lg p-6 text-center">
          <div className="text-4xl mb-4">✅</div>
          <h4 className="font-medium text-green-900 mb-2">No Missing Values Found</h4>
          <p className="text-sm text-green-800">Your dataset is complete with no missing values!</p>
          <div className="mt-4 text-sm text-green-700">
            <p>Total rows: {analysis.total_rows}</p>
            <p>Total columns: {analysis.total_columns}</p>
            <p>Complete cases: {analysis.pattern_analysis?.complete_cases_percentage?.toFixed(1)}%</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex justify-between items-start">
          <div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">
              🔍 Missing Values Analysis & Treatment
            </h3>
            <p className="text-gray-600">
              Comprehensive analysis of missing data patterns with intelligent treatment recommendations
            </p>
          </div>
          <div className="flex space-x-3">
            <button
              onClick={() => downloadOriginalMutation.mutate()}
              disabled={downloadOriginalMutation.isPending}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
            >
              <span>📥</span>
              <span>{downloadOriginalMutation.isPending ? 'Downloading...' : 'Download Original CSV'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6" aria-label="Tabs">
            {[
              { key: 'analysis', label: '📊 Analysis', icon: '📊' },
              { key: 'treatment', label: '🛠️ Treatment', icon: '🛠️' },
              { key: 'results', label: '📈 Results', icon: '📈' }
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
          {/* Analysis Tab */}
          {activeTab === 'analysis' && (
            <div className="space-y-6">
              {/* Summary Statistics */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                <div className="bg-blue-50 rounded-lg p-4">
                  <div className="text-2xl font-bold text-blue-600">
                    {analysis.missing_summary.total_missing_values}
                  </div>
                  <div className="text-sm text-blue-800">Total Missing Values</div>
                  <div className="text-xs text-blue-600 mt-1">
                    Out of {analysis.total_rows * analysis.total_columns} total cells
                  </div>
                </div>
                <div className="bg-red-50 rounded-lg p-4">
                  <div className="text-2xl font-bold text-red-600">
                    {analysis.missing_summary.columns_with_missing}
                  </div>
                  <div className="text-sm text-red-800">Columns Affected</div>
                  <div className="text-xs text-red-600 mt-1">
                    Out of {analysis.total_columns} total columns
                  </div>
                </div>
                <div className="bg-yellow-50 rounded-lg p-4">
                  <div className="text-2xl font-bold text-yellow-600">
                    {analysis.missing_summary.percentage_missing_overall.toFixed(1)}%
                  </div>
                  <div className="text-sm text-yellow-800">Overall Missing %</div>
                  <div className="text-xs text-yellow-600 mt-1">
                    Data completeness: {(100 - analysis.missing_summary.percentage_missing_overall).toFixed(1)}%
                  </div>
                </div>
                <div className="bg-green-50 rounded-lg p-4">
                  <div className="text-2xl font-bold text-green-600">
                    {analysis.pattern_analysis.complete_cases_percentage.toFixed(1)}%
                  </div>
                  <div className="text-sm text-green-800">Complete Cases</div>
                  <div className="text-xs text-green-600 mt-1">
                    {Math.round(analysis.pattern_analysis.complete_cases_percentage / 100 * analysis.total_rows)} rows have no missing values
                  </div>
                </div>
              </div>

              {/* Data Quality Assessment */}
              <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg p-4 mb-6 border border-indigo-200">
                <h4 className="font-medium text-indigo-900 mb-3">📊 Data Quality Assessment</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center">
                    <div className={`text-lg font-bold ${
                      analysis.missing_summary.percentage_missing_overall < 5 ? 'text-green-600' :
                      analysis.missing_summary.percentage_missing_overall < 20 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {analysis.missing_summary.percentage_missing_overall < 5 ? 'Excellent' :
                       analysis.missing_summary.percentage_missing_overall < 20 ? 'Good' : 'Poor'}
                    </div>
                    <div className="text-sm text-gray-600">Data Quality</div>
                  </div>
                  <div className="text-center">
                    <div className={`text-lg font-bold ${
                      analysis.pattern_analysis.complete_cases_percentage > 80 ? 'text-green-600' :
                      analysis.pattern_analysis.complete_cases_percentage > 50 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {analysis.pattern_analysis.complete_cases_percentage > 80 ? 'High' :
                       analysis.pattern_analysis.complete_cases_percentage > 50 ? 'Medium' : 'Low'}
                    </div>
                    <div className="text-sm text-gray-600">Usability</div>
                  </div>
                  <div className="text-center">
                    <div className={`text-lg font-bold ${
                      analysis.missing_summary.columns_with_missing / analysis.total_columns < 0.3 ? 'text-green-600' :
                      analysis.missing_summary.columns_with_missing / analysis.total_columns < 0.6 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {analysis.missing_summary.columns_with_missing / analysis.total_columns < 0.3 ? 'Minimal' :
                       analysis.missing_summary.columns_with_missing / analysis.total_columns < 0.6 ? 'Moderate' : 'Extensive'}
                    </div>
                    <div className="text-sm text-gray-600">Missing Spread</div>
                  </div>
                </div>
              </div>

              {/* Missing Values Chart */}
              <div className="bg-gray-50 rounded-lg p-4 mb-6">
                <h4 className="font-medium text-gray-900 mb-4">📊 Missing Values by Column</h4>
                <div className="space-y-3">
                  {missingColumns.map((column) => {
                    const colData = columnAnalysis[column];
                    const percentage = colData.missing_percentage;
                    return (
                      <div key={column} className="flex items-center space-x-3">
                        <div className="w-32 text-sm font-medium text-gray-700 truncate">
                          {column}
                        </div>
                        <div className="flex-1 bg-gray-200 rounded-full h-4 relative">
                          <div
                            className="bg-red-500 h-4 rounded-full transition-all duration-300"
                            style={{ width: `${Math.min(percentage, 100)}%` }}
                          ></div>
                          <div className="absolute inset-0 flex items-center justify-center text-xs font-medium text-gray-700">
                            {percentage.toFixed(1)}%
                          </div>
                        </div>
                        <div className="w-20 text-sm text-gray-600 text-right">
                          {colData.missing_count} / {analysis.total_rows}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Column-wise Analysis */}
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 mb-4">📋 Column-wise Missing Value Analysis</h4>
                <div className="space-y-4">
                  {missingColumns.map((column) => {
                    const colData = columnAnalysis[column];
                    return (
                      <div key={column} className="bg-white rounded-lg p-4 border border-gray-200">
                        <div className="flex justify-between items-start mb-3">
                          <div>
                            <h5 className="font-medium text-gray-900">{column}</h5>
                            <p className="text-sm text-gray-600">
                              {colData.data_type} • {colData.unique_values} unique values
                            </p>
                          </div>
                          <div className="text-right">
                            <div className="text-lg font-bold text-red-600">
                              {colData.missing_count} ({colData.missing_percentage.toFixed(1)}%)
                            </div>
                            <div className={`px-2 py-1 rounded text-xs font-medium ${getPatternColor(colData.pattern_type)}`}>
                              {colData.pattern_type}
                            </div>
                          </div>
                        </div>

                        {/* Missing Reasons */}
                        <div className="mb-3">
                          <h6 className="font-medium text-gray-800 mb-2">🤔 Possible Reasons:</h6>
                          <ul className="text-sm text-gray-700 space-y-1">
                            {colData.reasons.map((reason, idx) => (
                              <li key={idx} className="flex items-start">
                                <span className="text-gray-500 mr-2">•</span>
                                <span>{reason}</span>
                              </li>
                            ))}
                          </ul>
                        </div>

                        {/* Recommended Methods */}
                        <div>
                          <h6 className="font-medium text-gray-800 mb-2">💡 Recommended Treatment Methods:</h6>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            {colData.recommended_methods.slice(0, 4).map((method, idx) => (
                              <div key={idx} className="bg-gray-50 rounded p-3">
                                <div className="flex justify-between items-start mb-1">
                                  <span className="font-medium text-sm text-gray-900">
                                    {treatmentMethods.find(m => m.value === method.method)?.label || method.method}
                                  </span>
                                  <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs">
                                    Priority {method.priority}
                                  </span>
                                </div>
                                <p className="text-xs text-gray-600 mb-2">{method.reason}</p>
                                <div className="text-xs">
                                  <span className="text-green-600">✓ {method.pros[0]}</span>
                                  {method.cons[0] && (
                                    <span className="text-red-600 ml-2">✗ {method.cons[0]}</span>
                                  )}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Global Recommendations */}
              {analysis.recommendations.length > 0 && (
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

          {/* Treatment Tab */}
          {activeTab === 'treatment' && (
            <div className="space-y-6">
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <h4 className="font-medium text-yellow-900 mb-2">⚠️ Treatment Guidelines</h4>
                <ul className="text-sm text-yellow-800 space-y-1">
                  <li>• Drop rows if missing values are &lt; 5% of total data</li>
                  <li>• Use mean/median for numeric data, mode for categorical data</li>
                  <li>• KNN imputation works well for moderate missing percentages (&lt; 30%)</li>
                  <li>• Always preview treatment effects before applying</li>
                </ul>
              </div>

              {/* Method Selection */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Treatment Method
                  </label>
                  <select
                    value={selectedMethod}
                    onChange={(e) => setSelectedMethod(e.target.value)}
                    className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">Select a method...</option>
                    {treatmentMethods.map((method) => (
                      <option key={method.value} value={method.value}>
                        {method.label}
                      </option>
                    ))}
                  </select>
                  {selectedMethod && (
                    <p className="text-sm text-gray-600 mt-1">
                      {treatmentMethods.find(m => m.value === selectedMethod)?.description}
                    </p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Target Column (Optional)
                  </label>
                  <select
                    value={selectedColumn}
                    onChange={(e) => setSelectedColumn(e.target.value)}
                    className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">All columns</option>
                    {missingColumns.map((column) => (
                      <option key={column} value={column}>
                        {column} ({columnAnalysis[column].missing_percentage.toFixed(1)}% missing)
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Method-specific Options */}
              {selectedMethod === 'knn_imputation' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Number of Neighbors
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="20"
                    value={treatmentOptions.n_neighbors}
                    onChange={(e) => setTreatmentOptions(prev => ({
                      ...prev,
                      n_neighbors: parseInt(e.target.value)
                    }))}
                    className="w-32 border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              )}

              {selectedMethod === 'constant_imputation' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Constant Value
                  </label>
                  <input
                    type="text"
                    value={treatmentOptions.constant_value}
                    onChange={(e) => setTreatmentOptions(prev => ({
                      ...prev,
                      constant_value: e.target.value
                    }))}
                    className="w-48 border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="Enter constant value"
                  />
                </div>
              )}

              {selectedMethod === 'drop_columns' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Drop Threshold (% missing)
                  </label>
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.1"
                    value={treatmentOptions.threshold}
                    onChange={(e) => setTreatmentOptions(prev => ({
                      ...prev,
                      threshold: parseFloat(e.target.value)
                    }))}
                    className="w-32 border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <p className="text-sm text-gray-600 mt-1">
                    Columns with more than {(treatmentOptions.threshold! * 100).toFixed(0)}% missing values will be dropped
                  </p>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex space-x-4">
                <button
                  onClick={handlePreview}
                  disabled={!selectedMethod || previewMutation.isPending}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {previewMutation.isPending ? 'Previewing...' : '👁️ Preview Treatment'}
                </button>
                
                <button
                  onClick={handleTreatment}
                  disabled={!selectedMethod || treatmentMutation.isPending}
                  className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {treatmentMutation.isPending ? 'Applying...' : '✅ Apply Treatment'}
                </button>
              </div>

              {/* Preview Results */}
              {previewMutation.data && showPreview && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <h4 className="font-medium text-blue-900 mb-3">👁️ Treatment Preview</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <span className="font-medium">Current Shape:</span>
                      <div>{previewMutation.data.preview.current_shape[0]} × {previewMutation.data.preview.current_shape[1]}</div>
                    </div>
                    <div>
                      <span className="font-medium">Final Shape:</span>
                      <div>{previewMutation.data.preview.final_shape[0]} × {previewMutation.data.preview.final_shape[1]}</div>
                    </div>
                    <div>
                      <span className="font-medium">Current Missing:</span>
                      <div>{previewMutation.data.preview.current_missing}</div>
                    </div>
                    <div>
                      <span className="font-medium">After Treatment:</span>
                      <div>{previewMutation.data.preview.missing_after_treatment || 0}</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Results Tab */}
          {activeTab === 'results' && (
            <div className="space-y-6">
              {treatmentMutation.data ? (
                <>
                  <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-6">
                    <h4 className="font-medium text-green-900 mb-3">✅ Treatment Applied Successfully</h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <span className="font-medium">Method Used:</span>
                        <div>{treatmentMutation.data.treatment_info.method}</div>
                      </div>
                      <div>
                        <span className="font-medium">Original Shape:</span>
                        <div>{treatmentMutation.data.treatment_info.original_shape[0]} × {treatmentMutation.data.treatment_info.original_shape[1]}</div>
                      </div>
                      <div>
                        <span className="font-medium">Final Shape:</span>
                        <div>{treatmentMutation.data.treatment_info.final_shape[0]} × {treatmentMutation.data.treatment_info.final_shape[1]}</div>
                      </div>
                      <div>
                        <span className="font-medium">Missing Values Remaining:</span>
                        <div>{treatmentMutation.data.treatment_info.missing_values_remaining}</div>
                      </div>
                    </div>
                  </div>

                  {/* Treatment Method Explanation */}
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
                    <h4 className="font-medium text-blue-900 mb-3">🔬 Treatment Method Explanation</h4>
                    <div className="space-y-3">
                      <div>
                        <span className="font-medium text-blue-800">Method:</span>
                        <span className="ml-2">
                          {treatmentMethods.find(m => m.value === treatmentMutation.data.treatment_info.method)?.label || treatmentMutation.data.treatment_info.method}
                        </span>
                      </div>
                      <div>
                        <span className="font-medium text-blue-800">Description:</span>
                        <span className="ml-2">
                          {treatmentMethods.find(m => m.value === treatmentMutation.data.treatment_info.method)?.description || 'Custom treatment method applied'}
                        </span>
                      </div>
                      <div>
                        <span className="font-medium text-blue-800">Why this method was chosen:</span>
                        <div className="ml-2 mt-1 text-sm">
                          {(() => {
                            const method = treatmentMutation.data.treatment_info.method;
                            const originalMissing = treatmentMutation.data.treatment_info.original_shape[0] * treatmentMutation.data.treatment_info.original_shape[1] - 
                                                  (treatmentMutation.data.treatment_info.original_shape[0] * treatmentMutation.data.treatment_info.original_shape[1] - 
                                                   treatmentMutation.data.treatment_info.missing_values_remaining);
                            const missingPercentage = (originalMissing / (treatmentMutation.data.treatment_info.original_shape[0] * treatmentMutation.data.treatment_info.original_shape[1])) * 100;
                            
                            if (method === 'drop_rows') {
                              return `Row deletion was chosen because the missing data percentage was low (${missingPercentage.toFixed(1)}%), making it safe to remove incomplete rows without significant data loss.`;
                            } else if (method === 'mean_imputation') {
                              return `Mean imputation was selected for numerical data to preserve the central tendency while filling missing values with statistically reasonable estimates.`;
                            } else if (method === 'median_imputation') {
                              return `Median imputation was chosen for numerical data as it's more robust to outliers and works well with skewed distributions.`;
                            } else if (method === 'mode_imputation') {
                              return `Mode imputation was selected for categorical data to fill missing values with the most frequently occurring category.`;
                            } else if (method === 'knn_imputation') {
                              return `KNN imputation was chosen to leverage relationships between similar records, providing more accurate estimates than simple statistical methods.`;
                            } else if (method === 'forward_fill') {
                              return `Forward fill was selected for time-series data to maintain temporal continuity by propagating the last known value.`;
                            } else {
                              return `This method was selected based on the data characteristics and missing value patterns to optimize data quality and preserve information.`;
                            }
                          })()}
                        </div>
                      </div>
                      {treatmentMutation.data.treatment_info.parameters && Object.keys(treatmentMutation.data.treatment_info.parameters).length > 0 && (
                        <div>
                          <span className="font-medium text-blue-800">Parameters used:</span>
                          <div className="ml-2 mt-1 text-sm">
                            {Object.entries(treatmentMutation.data.treatment_info.parameters).map(([key, value]) => (
                              <div key={key}>• {key}: {String(value)}</div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="flex justify-center space-x-4">
                    <button
                      onClick={() => downloadOriginalMutation.mutate()}
                      disabled={downloadOriginalMutation.isPending}
                      className="px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
                    >
                      <span>📥</span>
                      <span>{downloadOriginalMutation.isPending ? 'Downloading...' : 'Download Original Dataset (CSV)'}</span>
                    </button>
                    <button
                      onClick={() => downloadTreatedMutation.mutate()}
                      disabled={downloadTreatedMutation.isPending}
                      className="px-6 py-3 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
                    >
                      <span>📥</span>
                      <span>{downloadTreatedMutation.isPending ? 'Downloading...' : 'Download Cleaned Dataset (CSV)'}</span>
                    </button>
                  </div>
                </>
              ) : (
                <div className="text-center py-12 text-gray-500">
                  <div className="text-4xl mb-4">🛠️</div>
                  <p>No treatment has been applied yet.</p>
                  <p className="text-sm">Go to the Treatment tab to apply missing value treatments.</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MissingValuesInsights;
