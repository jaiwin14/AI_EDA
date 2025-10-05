import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import aiService from '../../services/aiService';

interface MissingValuesInsightsProps {
  datasetId: string;
  onTreatmentComplete?: (treatedDatasetId: string) => void;
  onSkipToOutliers?: () => void;
}

const MissingValuesInsights: React.FC<MissingValuesInsightsProps> = ({ 
  datasetId, 
  onTreatmentComplete,
  onSkipToOutliers 
}) => {
  const [activeTab, setActiveTab] = useState<'analysis' | 'treatment' | 'results'>('analysis');
  const [selectedMethod, setSelectedMethod] = useState<string>('');
  const [selectedColumn, setSelectedColumn] = useState<string>('');
  const [nNeighbors, setNNeighbors] = useState<number>(5);
  const [constantValue, setConstantValue] = useState<string>('0');
  const [threshold, setThreshold] = useState<number>(0.5);
  const [showPreview, setShowPreview] = useState(false);

  const queryClient = useQueryClient();

  // Check treatment status
  const { data: statusData, refetch: refetchStatus } = useQuery({
    queryKey: ['treatment-status', datasetId],
    queryFn: () => aiService.getTreatmentStatus(datasetId),
    refetchInterval: 3000,
  });

  // Fetch missing values analysis
  const { data: analysisData, isLoading, error } = useQuery({
    queryKey: ['missing-values-analysis', datasetId],
    queryFn: () => aiService.getMissingValuesInsights(datasetId),
    enabled: !!datasetId,
  });

  // Preview mutation
  const previewMutation = useMutation({
    mutationFn: () => aiService.previewMissingValuesTreatment(datasetId, selectedMethod, {
      column: selectedColumn || undefined,
      n_neighbors: nNeighbors,
      constant_value: constantValue,
      threshold: threshold
    }),
    onSuccess: () => {
      console.log('✅ Preview generated');
      setShowPreview(true);
    },
    onError: (error) => {
      console.error('❌ Preview failed:', error);
    }
  });

  // Treatment mutation - FIXED
  const treatmentMutation = useMutation({
    mutationFn: async () => {
      console.log('🔧 Starting treatment...');
      console.log('Dataset ID:', datasetId);
      console.log('Method:', selectedMethod);
      console.log('Column:', selectedColumn);
      console.log('Parameters:', { nNeighbors, constantValue, threshold });
      
      const result = await aiService.treatMissingValues(datasetId, selectedMethod, {
        column: selectedColumn || undefined,
        n_neighbors: nNeighbors,
        constant_value: constantValue,
        threshold: threshold
      });
      
      console.log('✅ Treatment response:', result);
      return result;
    },
    onSuccess: async (response) => {
      console.log('✅ Treatment applied successfully:', response);
      
      // Invalidate queries to refresh data
      await queryClient.invalidateQueries({ queryKey: ['missing-values-analysis'] });
      await queryClient.invalidateQueries({ queryKey: ['treatment-status'] });
      
      // Refetch status
      await refetchStatus();
      
      // Switch to results tab
      setActiveTab('results');
      
      // Call the completion callback
      if (onTreatmentComplete && response.treated_dataset_id) {
        console.log('📤 Calling onTreatmentComplete with:', response.treated_dataset_id);
        setTimeout(() => {
          onTreatmentComplete(response.treated_dataset_id);
        }, 1000);
      }
    },
    onError: (error: any) => {
      console.error('❌ Treatment failed:', error);
      const errorMessage = error?.response?.data?.detail || error.message || 'Unknown error occurred';
      alert(`Treatment failed: ${errorMessage}`);
    }
  });

  // Download mutations
  const downloadOriginal = useMutation({
    mutationFn: () => aiService.downloadOriginalDataset(datasetId),
    onSuccess: (blob) => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${datasetId}_original.csv`;
      a.click();
      window.URL.revokeObjectURL(url);
    },
  });

  const downloadTreated = useMutation({
    mutationFn: () => aiService.downloadTreatedDataset(datasetId),
    onSuccess: (blob) => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${datasetId}_cleaned.csv`;
      a.click();
      window.URL.revokeObjectURL(url);
    },
  });

  const analysis = analysisData?.data;
  const columnAnalysis = analysis?.column_analysis || {};
  const missingColumns = Object.keys(columnAnalysis);
  const hasMissing = missingColumns.length > 0;

  // Auto-skip if no missing values
  useEffect(() => {
    if (!isLoading && !hasMissing && onSkipToOutliers) {
      console.log('✅ No missing values detected, auto-skipping to outliers');
      const timer = setTimeout(() => {
        onSkipToOutliers();
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [isLoading, hasMissing, onSkipToOutliers]);

  // Handle treatment completion
  useEffect(() => {
    if (statusData?.treatment_applied && !treatmentMutation.data) {
      console.log('✅ Treatment already applied, dataset ready');
      if (onTreatmentComplete && statusData.treated_dataset_id) {
        console.log('📤 Auto-calling onTreatmentComplete');
        setTimeout(() => {
          if (statusData.treated_dataset_id) {
            onTreatmentComplete(statusData.treated_dataset_id);
          }
        }, 1000);
      }
    }
  }, [statusData, onTreatmentComplete]);

  const treatmentMethods = [
    { value: 'drop_rows', label: '🗑️ Drop Rows', description: 'Remove rows with missing values' },
    { value: 'drop_columns', label: '🗑️ Drop Columns', description: 'Remove columns with high missing %' },
    { value: 'mean_imputation', label: '📊 Mean Imputation', description: 'Fill with mean (numeric)' },
    { value: 'median_imputation', label: '📊 Median Imputation', description: 'Fill with median (numeric)' },
    { value: 'mode_imputation', label: '📊 Mode Imputation', description: 'Fill with most frequent value' },
    { value: 'forward_fill', label: '⏭️ Forward Fill', description: 'Fill with previous value' },
    { value: 'backward_fill', label: '⏮️ Backward Fill', description: 'Fill with next value' },
    { value: 'knn_imputation', label: '🔍 KNN Imputation', description: 'Fill using similar neighbors' },
    { value: 'constant_imputation', label: '🔢 Constant Fill', description: 'Fill with constant value' },
  ];

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-3 text-gray-600">Analyzing missing values...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <h4 className="font-medium text-red-900 mb-2">Error Loading Analysis</h4>
        <p className="text-sm text-red-800">Unable to analyze missing values. Please try again.</p>
      </div>
    );
  }

  // No missing values - auto-skip
  if (!hasMissing) {
    return (
      <div className="space-y-6">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-2">
            🔍 Missing Values Analysis
          </h3>
          <p className="text-gray-600">Checking for missing values...</p>
        </div>
        
        <div className="bg-green-50 border border-green-200 rounded-lg p-8 text-center">
          <div className="text-6xl mb-4">✅</div>
          <h4 className="text-lg font-semibold text-green-900 mb-2">No Missing Values Found!</h4>
          <p className="text-sm text-green-800 mb-4">Your dataset is complete. Proceeding to Outlier Detection...</p>
          <button
            onClick={onSkipToOutliers}
            className="mt-4 px-6 py-3 bg-green-600 text-white rounded-md hover:bg-green-700 font-medium"
          >
            Continue to Outlier Detection →
          </button>
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
              Comprehensive analysis and intelligent treatment recommendations
            </p>
            {statusData?.treatment_applied && (
              <div className="mt-3 inline-flex items-center px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm">
                <span className="mr-2">✅</span>
                <span>Treatment Applied - Dataset Cleaned</span>
              </div>
            )}
          </div>
          <button
            onClick={() => downloadOriginal.mutate()}
            disabled={downloadOriginal.isPending}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            {downloadOriginal.isPending ? 'Downloading...' : '📥 Download Original'}
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6">
            {['analysis', 'treatment', 'results'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab as any)}
                className={`py-4 px-1 border-b-2 font-medium text-sm capitalize ${
                  activeTab === tab
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                {tab === 'analysis' ? '📊' : tab === 'treatment' ? '🛠️' : '📈'} {tab}
              </button>
            ))}
          </nav>
        </div>

        <div className="p-6">
          {/* Analysis Tab */}
          {activeTab === 'analysis' && (
            <div className="space-y-6">
              {/* Summary Cards */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-blue-50 rounded-lg p-4">
                  <div className="text-2xl font-bold text-blue-600">
                    {analysis?.missing_summary?.total_missing_values || 0}
                  </div>
                  <div className="text-sm text-blue-800">Total Missing</div>
                </div>
                <div className="bg-red-50 rounded-lg p-4">
                  <div className="text-2xl font-bold text-red-600">
                    {analysis?.missing_summary?.columns_with_missing || 0}
                  </div>
                  <div className="text-sm text-red-800">Columns Affected</div>
                </div>
                <div className="bg-yellow-50 rounded-lg p-4">
                  <div className="text-2xl font-bold text-yellow-600">
                    {analysis?.missing_summary?.percentage_missing_overall?.toFixed(1) || 0}%
                  </div>
                  <div className="text-sm text-yellow-800">Overall Missing</div>
                </div>
                <div className="bg-green-50 rounded-lg p-4">
                  <div className="text-2xl font-bold text-green-600">
                    {analysis?.pattern_analysis?.complete_cases_percentage?.toFixed(1) || 0}%
                  </div>
                  <div className="text-sm text-green-800">Complete Cases</div>
                </div>
              </div>

              {/* Recommendations */}
              {analysis?.recommendations && analysis.recommendations.length > 0 && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <h4 className="font-medium text-blue-900 mb-3">💡 AI Recommendations</h4>
                  <ul className="space-y-2">
                    {analysis.recommendations.map((rec: string, idx: number) => (
                      <li key={idx} className="text-sm text-blue-800 flex items-start">
                        <span className="mr-2">•</span>
                        <span>{rec}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Column List */}
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 mb-4">📋 Columns with Missing Values</h4>
                <div className="space-y-3">
                  {missingColumns.map((column) => {
                    const colData = columnAnalysis[column];
                    return (
                      <div key={column} className="bg-white rounded-lg p-4 border">
                        <div className="flex justify-between items-center mb-2">
                          <span className="font-medium text-gray-900">{column}</span>
                          <span className="text-lg font-bold text-red-600">
                            {colData.missing_percentage?.toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
                          <div
                            className="bg-red-500 h-2 rounded-full"
                            style={{ width: `${Math.min(colData.missing_percentage, 100)}%` }}
                          />
                        </div>
                        <div className="text-xs text-gray-600 mb-2">
                          {colData.missing_count} missing out of {analysis?.total_rows} rows
                        </div>
                        {colData.recommended_methods && colData.recommended_methods.length > 0 && (
                          <div className="mt-3 pt-3 border-t">
                            <div className="text-xs font-medium text-gray-700 mb-2">Recommended Methods:</div>
                            <div className="space-y-1">
                              {colData.recommended_methods.slice(0, 2).map((method: any, idx: number) => (
                                <div key={idx} className="text-xs text-gray-600 bg-gray-50 rounded px-2 py-1">
                                  <span className="font-medium">{method.method}</span>: {method.reason}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Action Button */}
              <div className="text-center">
                <button
                  onClick={() => setActiveTab('treatment')}
                  className="px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 font-medium"
                >
                  Proceed to Treatment →
                </button>
              </div>
            </div>
          )}

          {/* Treatment Tab */}
          {activeTab === 'treatment' && (
            <div className="space-y-6">
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <h4 className="font-medium text-yellow-900 mb-2">⚠️ Treatment Guidelines</h4>
                <ul className="text-sm text-yellow-800 space-y-1">
                  <li>• Preview treatment before applying to see the effects</li>
                  <li>• Treatment creates a new cleaned dataset (original preserved)</li>
                  <li>• Choose method based on your data type and analysis needs</li>
                  <li>• You can download both original and cleaned datasets</li>
                </ul>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Treatment Method *
                  </label>
                  <select
                    value={selectedMethod}
                    onChange={(e) => {
                      setSelectedMethod(e.target.value);
                      setShowPreview(false);
                    }}
                    className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="">Select method...</option>
                    {treatmentMethods.map((m) => (
                      <option key={m.value} value={m.value}>{m.label}</option>
                    ))}
                  </select>
                  {selectedMethod && (
                    <p className="text-sm text-gray-600 mt-2">
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
                    onChange={(e) => {
                      setSelectedColumn(e.target.value);
                      setShowPreview(false);
                    }}
                    className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="">All columns</option>
                    {missingColumns.map((col) => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                  <p className="text-sm text-gray-500 mt-1">
                    Leave empty to apply to all columns
                  </p>
                </div>
              </div>

              {/* Method-specific options */}
              {selectedMethod === 'knn_imputation' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Number of Neighbors
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="20"
                    value={nNeighbors}
                    onChange={(e) => {
                      setNNeighbors(parseInt(e.target.value));
                      setShowPreview(false);
                    }}
                    className="w-32 border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500"
                  />
                  <p className="text-sm text-gray-500 mt-1">
                    How many similar rows to use for imputation (default: 5)
                  </p>
                </div>
              )}

              {selectedMethod === 'constant_imputation' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Constant Value
                  </label>
                  <input
                    type="text"
                    value={constantValue}
                    onChange={(e) => {
                      setConstantValue(e.target.value);
                      setShowPreview(false);
                    }}
                    className="w-48 border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500"
                    placeholder="Enter value (e.g., 0, Unknown)"
                  />
                  <p className="text-sm text-gray-500 mt-1">
                    Value to fill missing entries with
                  </p>
                </div>
              )}

              {selectedMethod === 'drop_columns' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Drop Threshold (0-1)
                  </label>
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.1"
                    value={threshold}
                    onChange={(e) => {
                      setThreshold(parseFloat(e.target.value));
                      setShowPreview(false);
                    }}
                    className="w-32 border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500"
                  />
                  <p className="text-sm text-gray-500 mt-1">
                    Drop columns with missing percentage above this threshold (default: 0.5 = 50%)
                  </p>
                </div>
              )}

              {/* Preview Results */}
              {showPreview && previewMutation.data && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <h4 className="font-medium text-blue-900 mb-3">👁️ Treatment Preview</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <span className="font-medium text-blue-800">Current Shape:</span>
                      <div className="text-blue-900">
                        {previewMutation.data.preview?.current_shape?.join(' × ')}
                      </div>
                    </div>
                    <div>
                      <span className="font-medium text-blue-800">After Treatment:</span>
                      <div className="text-blue-900">
                        {previewMutation.data.preview?.final_shape?.join(' × ')}
                      </div>
                    </div>
                    <div>
                      <span className="font-medium text-blue-800">Rows to Remove:</span>
                      <div className="text-blue-900">
                        {previewMutation.data.preview?.rows_to_remove || 0}
                      </div>
                    </div>
                    <div>
                      <span className="font-medium text-blue-800">Columns to Remove:</span>
                      <div className="text-blue-900">
                        {previewMutation.data.preview?.columns_to_remove?.length || 0}
                      </div>
                    </div>
                  </div>
                  {previewMutation.data.preview?.columns_to_remove && 
                   previewMutation.data.preview.columns_to_remove.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-blue-200">
                      <span className="font-medium text-blue-800 text-sm">Columns to be removed:</span>
                      <div className="text-sm text-blue-900 mt-1">
                        {previewMutation.data.preview.columns_to_remove.join(', ')}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex space-x-4">
                <button
                  onClick={() => previewMutation.mutate()}
                  disabled={!selectedMethod || previewMutation.isPending}
                  className="px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
                >
                  {previewMutation.isPending ? (
                    <span className="flex items-center">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Previewing...
                    </span>
                  ) : '👁️ Preview Treatment'}
                </button>
                
                <button
                  onClick={() => {
                    if (window.confirm('Apply this treatment to the dataset? This will create a new cleaned dataset.')) {
                      treatmentMutation.mutate();
                    }
                  }}
                  disabled={!selectedMethod || treatmentMutation.isPending}
                  className="px-6 py-3 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
                >
                  {treatmentMutation.isPending ? (
                    <span className="flex items-center">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Applying...
                    </span>
                  ) : '✅ Apply Treatment'}
                </button>
              </div>

              {treatmentMutation.isError && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <h4 className="font-medium text-red-900 mb-2">❌ Treatment Failed</h4>
                  <p className="text-sm text-red-800">
                    {treatmentMutation.error instanceof Error ? treatmentMutation.error.message : 'Unknown error occurred'}
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Results Tab */}
          {activeTab === 'results' && (
            <div className="space-y-6">
              {treatmentMutation.data || statusData?.treatment_applied ? (
                <>
                  <div className="bg-green-50 border border-green-200 rounded-lg p-6">
                    <div className="flex items-center mb-4">
                      <div className="text-4xl mr-4">✅</div>
                      <div>
                        <h4 className="text-lg font-semibold text-green-900">Treatment Applied Successfully!</h4>
                        <p className="text-sm text-green-800">Your dataset has been cleaned and is ready for the next step.</p>
                      </div>
                    </div>
                    
                    {treatmentMutation.data?.treatment_info && (
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4 pt-4 border-t border-green-200">
                        <div>
                          <span className="text-sm font-medium text-green-800">Method:</span>
                          <div className="text-green-900 font-medium">
                            {treatmentMethods.find(m => m.value === treatmentMutation.data.treatment_info?.method)?.label}
                          </div>
                        </div>
                        <div>
                          <span className="text-sm font-medium text-green-800">Original:</span>
                          <div className="text-green-900 font-medium">
                            {treatmentMutation.data.treatment_info?.original_shape?.join(' × ')}
                          </div>
                        </div>
                        <div>
                          <span className="text-sm font-medium text-green-800">Final:</span>
                          <div className="text-green-900 font-medium">
                            {treatmentMutation.data.treatment_info?.final_shape?.join(' × ')}
                          </div>
                        </div>
                        <div>
                          <span className="text-sm font-medium text-green-800">Missing Remaining:</span>
                          <div className="text-green-900 font-medium">
                            {treatmentMutation.data.treatment_info?.missing_values_remaining || 0}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Download and Continue Buttons */}
                  <div className="flex flex-col space-y-4">
                    <div className="flex justify-center space-x-4">
                      <button
                        onClick={() => downloadOriginal.mutate()}
                        disabled={downloadOriginal.isPending}
                        className="px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 font-medium"
                      >
                        {downloadOriginal.isPending ? 'Downloading...' : '📥 Download Original Dataset'}
                      </button>
                      <button
                        onClick={() => downloadTreated.mutate()}
                        disabled={downloadTreated.isPending}
                        className="px-6 py-3 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 font-medium"
                      >
                        {downloadTreated.isPending ? 'Downloading...' : '📥 Download Cleaned Dataset'}
                      </button>
                    </div>

                    {/* Continue Button */}
                    {statusData?.can_proceed_to_outliers && onSkipToOutliers && (
                      <div className="text-center pt-4 border-t">
                        <p className="text-sm text-gray-600 mb-3">
                          Dataset is ready for outlier detection
                        </p>
                        <button
                          onClick={() => {
                            console.log('🚀 Navigating to outliers');
                            onSkipToOutliers();
                          }}
                          className="px-8 py-4 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 font-medium text-lg shadow-lg hover:shadow-xl transition-all"
                        >
                          Continue to Outlier Detection →
                        </button>
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <div className="text-center py-12">
                  <div className="text-6xl mb-4">🛠️</div>
                  <h4 className="text-lg font-semibold text-gray-700 mb-2">No Treatment Applied Yet</h4>
                  <p className="text-sm text-gray-600 mb-4">
                    Go to the Treatment tab to clean your data
                  </p>
                  <button
                    onClick={() => setActiveTab('treatment')}
                    className="px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 font-medium"
                  >
                    Go to Treatment →
                  </button>
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