import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import aiService from '../../services/aiService';
import './MissingValuesInsights.css';

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
      <div className="loading-container">
        <div className="spinner"></div>
        <span>Analyzing missing values...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-box">
        <h4>Error Loading Analysis</h4>
        <p>Unable to analyze missing values. Please try again.</p>
      </div>
    );
  }

  // No missing values - auto-skip
  if (!hasMissing) {
    return (
      <div className="mv-container">
        <div className="mv-tabs">
          <h3 className="mv-title">
            🔍 Missing Values Analysis
          </h3>
          <p className="mv-description">Checking for missing values...</p>
        </div>

        <div className="mv-results-box">
          <div className="text-6xl mb-4">✅</div>
          <h4>No Missing Values Found!</h4>
          <p>Your dataset is complete. Proceeding to Outlier Detection...</p>
          <button
            onClick={onSkipToOutliers}
            className="mv-btn mv-btn-secondary"
          >
            Continue to Outlier Detection →
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="mv-container">
      {/* Header */}
      <div className="mv-tabs">
        <div>
          <div>
            <h3 className="mv-title">
              🔍 Missing Values Analysis & Treatment
            </h3>
            <p className="mv-description">
              Comprehensive analysis and intelligent treatment recommendations
            </p>
            {statusData?.treatment_applied && (
              <div className="tag-green">
                <span>✅</span>
                <span>Treatment Applied - Dataset Cleaned</span>
              </div>
            )}
          </div>
          <button
            onClick={() => downloadOriginal.mutate()}
            disabled={downloadOriginal.isPending}
            className="mv-btn mv-btn-primary"
          >
            {downloadOriginal.isPending ? 'Downloading...' : '📥 Download Original'}
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="mv-tabs">
        <div className="mv-tab-nav">
          <nav>
            {['analysis', 'treatment', 'results'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab as any)}
                className={`mv-tab-btn ${activeTab === tab ? 'active' : ''}`}
              >
                {tab === 'analysis' ? '📊' : tab === 'treatment' ? '🛠️' : '📈'} {tab}
              </button>
            ))}
          </nav>
        </div>

        <div className="mv-tab-content">
          {/* Analysis Tab */}
          {activeTab === 'analysis' && (
            <div className="mv-container">
              {/* Summary Cards */}
              <div className="mv-summary-grid">
                <div className="mv-summary-card total">
                  <div>
                    {analysis?.missing_summary?.total_missing_values || 0}
                  </div>
                  <div>Total Missing</div>
                </div>
                <div className="mv-summary-card affected">
                  <div>
                    {analysis?.missing_summary?.columns_with_missing || 0}
                  </div>
                  <div>Columns Affected</div>
                </div>
                <div className="mv-summary-card percentage">
                  <div>
                    {analysis?.missing_summary?.percentage_missing_overall?.toFixed(1) || 0}%
                  </div>
                  <div>Overall Missing</div>
                </div>
                <div className="mv-summary-card complete">
                  <div>
                    {analysis?.pattern_analysis?.complete_cases_percentage?.toFixed(1) || 0}%
                  </div>
                  <div>Complete Cases</div>
                </div>
              </div>

              {/* Recommendations */}
              {analysis?.recommendations && analysis.recommendations.length > 0 && (
                <div className="mv-recommendations">
                  <h4 className="mv-recommendations-title">💡 AI Recommendations</h4>
                  <ul>
                    {analysis.recommendations.map((rec: string, idx: number) => (
                      <li key={idx}>
                        <span>•</span>
                        <span>{rec}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Column List */}
              <div className="mv-column-list">
                <h4>📋 Columns with Missing Values</h4>
                <div>
                  {missingColumns.map((column) => {
                    const colData = columnAnalysis[column];
                    return (
                      <div key={column} className="mv-column-item">
                        <div>
                          <span>{column}</span>
                          <span>
                            {colData.missing_percentage?.toFixed(1)}%
                          </span>
                        </div>
                        <div className="mv-progress-bar">
                          <div
                            className="mv-progress"
                            style={{ width: `${Math.min(colData.missing_percentage, 100)}%` }}
                          />
                        </div>
                        <div>
                          {colData.missing_count} missing out of {analysis?.total_rows} rows
                        </div>
                        {colData.recommended_methods && colData.recommended_methods.length > 0 && (
                          <div>
                            <div>Recommended Methods:</div>
                            <div>
                              {colData.recommended_methods.slice(0, 2).map((method: any, idx: number) => (
                                <div key={idx}>
                                  <span>{method.method}</span>: {method.reason}
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
              <div>
                <button
                  onClick={() => setActiveTab('treatment')}
                  className="mv-btn mv-btn-primary"
                >
                  Proceed to Treatment →
                </button>
              </div>
            </div>
          )}

          {/* Treatment Tab */}
          {activeTab === 'treatment' && (
            <div className="mv-treatment-section">
              <div className="mv-treatment-guidelines">
                <h4>⚠️ Treatment Guidelines</h4>
                <ul>
                  <li>• Preview treatment before applying to see the effects</li>
                  <li>• Treatment creates a new cleaned dataset (original preserved)</li>
                  <li>• Choose method based on your data type and analysis needs</li>
                  <li>• You can download both original and cleaned datasets</li>
                </ul>
              </div>

              <div className="mv-form-grid">
                <div className="mv-form-group">
                  <label>
                    Treatment Method *
                  </label>
                  <select
                    value={selectedMethod}
                    onChange={(e) => {
                      setSelectedMethod(e.target.value);
                      setShowPreview(false);
                    }}
                  >
                    <option value="">Select method...</option>
                    {treatmentMethods.map((m) => (
                      <option key={m.value} value={m.value}>{m.label}</option>
                    ))}
                  </select>
                  {selectedMethod && (
                    <p>
                      {treatmentMethods.find(m => m.value === selectedMethod)?.description}
                    </p>
                  )}
                </div>

                <div className="mv-form-group">
                  <label>
                    Target Column (Optional)
                  </label>
                  <select
                    value={selectedColumn}
                    onChange={(e) => {
                      setSelectedColumn(e.target.value);
                      setShowPreview(false);
                    }}
                  >
                    <option value="">All columns</option>
                    {missingColumns.map((col) => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                  <p>
                    Leave empty to apply to all columns
                  </p>
                </div>
              </div>

              {/* Method-specific options */}
              {selectedMethod === 'knn_imputation' && (
                <div className="mv-form-group">
                  <label>
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
                  />
                  <p>
                    How many similar rows to use for imputation (default: 5)
                  </p>
                </div>
              )}

              {selectedMethod === 'constant_imputation' && (
                <div className="mv-form-group">
                  <label>
                    Constant Value
                  </label>
                  <input
                    type="text"
                    value={constantValue}
                    onChange={(e) => {
                      setConstantValue(e.target.value);
                      setShowPreview(false);
                    }}
                    placeholder="Enter value (e.g., 0, Unknown)"
                  />
                  <p>
                    Value to fill missing entries with
                  </p>
                </div>
              )}

              {selectedMethod === 'drop_columns' && (
                <div className="mv-form-group">
                  <label>
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
                  />
                  <p>
                    Drop columns with missing percentage above this threshold (default: 0.5 = 50%)
                  </p>
                </div>
              )}

              {/* Preview Results */}
              {showPreview && previewMutation.data && (
                <div className="mv-preview-box">
                  <h4>👁️ Treatment Preview</h4>
                  <div>
                    <div>
                      <span>Current Shape:</span>
                      <div>
                        {previewMutation.data.preview?.current_shape?.join(' × ')}
                      </div>
                    </div>
                    <div>
                      <span>After Treatment:</span>
                      <div>
                        {previewMutation.data.preview?.final_shape?.join(' × ')}
                      </div>
                    </div>
                    <div>
                      <span>Rows to Remove:</span>
                      <div>
                        {previewMutation.data.preview?.rows_to_remove || 0}
                      </div>
                    </div>
                    <div>
                      <span>Columns to Remove:</span>
                      <div>
                        {previewMutation.data.preview?.columns_to_remove?.length || 0}
                      </div>
                    </div>
                  </div>
                  {previewMutation.data.preview?.columns_to_remove &&
                    previewMutation.data.preview.columns_to_remove.length > 0 && (
                    <div>
                      <span>Columns to be removed:</span>
                      <div>
                        {previewMutation.data.preview.columns_to_remove.join(', ')}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Action Buttons */}
              <div className="mv-action-buttons">
                <button
                  onClick={() => previewMutation.mutate()}
                  disabled={!selectedMethod || previewMutation.isPending}
                  className="mv-btn mv-btn-primary"
                >
                  {previewMutation.isPending ? (
                    <span>
                      <div className="spinner"></div>
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
                  className="mv-btn mv-btn-secondary"
                >
                  {treatmentMutation.isPending ? (
                    <span>
                      <div className="spinner"></div>
                      Applying...
                    </span>
                  ) : '✅ Apply Treatment'}
                </button>
              </div>

              {treatmentMutation.isError && (
                <div className="error-box">
                  <h4>❌ Treatment Failed</h4>
                  <p>
                    {treatmentMutation.error instanceof Error ? treatmentMutation.error.message : 'Unknown error occurred'}
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Results Tab */}
          {activeTab === 'results' && (
            <div className="mv-container">
              {treatmentMutation.data || statusData?.treatment_applied ? (
                <>
                  <div className="mv-results-box">
                    <div>
                      <div>✅</div>
                      <div>
                        <h4>Treatment Applied Successfully!</h4>
                        <p>Your dataset has been cleaned and is ready for the next step.</p>
                      </div>
                    </div>

                    {treatmentMutation.data?.treatment_info && (
                      <div>
                        <div>
                          <span>Method:</span>
                          <div>
                            {treatmentMethods.find(m => m.value === treatmentMutation.data.treatment_info?.method)?.label}
                          </div>
                        </div>
                        <div>
                          <span>Original:</span>
                          <div>
                            {treatmentMutation.data.treatment_info?.original_shape?.join(' × ')}
                          </div>
                        </div>
                        <div>
                          <span>Final:</span>
                          <div>
                            {treatmentMutation.data.treatment_info?.final_shape?.join(' × ')}
                          </div>
                        </div>
                        <div>
                          <span>Missing Remaining:</span>
                          <div>
                            {treatmentMutation.data.treatment_info?.missing_values_remaining || 0}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Download and Continue Buttons */}
                  <div>
                    <div>
                      <button
                        onClick={() => downloadOriginal.mutate()}
                        disabled={downloadOriginal.isPending}
                        className="mv-btn mv-btn-primary"
                      >
                        {downloadOriginal.isPending ? 'Downloading...' : '📥 Download Original Dataset'}
                      </button>
                      <button
                        onClick={() => downloadTreated.mutate()}
                        disabled={downloadTreated.isPending}
                        className="mv-btn mv-btn-secondary"
                      >
                        {downloadTreated.isPending ? 'Downloading...' : '📥 Download Cleaned Dataset'}
                      </button>
                    </div>

                    {/* Continue Button */}
                    {statusData?.can_proceed_to_outliers && onSkipToOutliers && (
                      <div>
                        <p>
                          Dataset is ready for outlier detection
                        </p>
                        <button
                          onClick={() => {
                            console.log('🚀 Navigating to outliers');
                            onSkipToOutliers();
                          }}
                          className="mv-btn mv-btn-primary"
                        >
                          Continue to Outlier Detection →
                        </button>
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <div>
                  <div>🛠️</div>
                  <h4>No Treatment Applied Yet</h4>
                  <p>
                    Go to the Treatment tab to clean your data
                  </p>
                  <button
                    onClick={() => setActiveTab('treatment')}
                    className="mv-btn mv-btn-primary"
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