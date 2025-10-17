import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import aiService from '../../services/aiService';
import './OutlierAnalysisSection.css';

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
      <div className="outlier-empty">
        <div className="outlier-spinner"></div>
        <span>Analyzing outliers...</span>
      </div>
    );
  }

  if (analysisError) {
    return (
      <div className="outlier-error">
        <h4>Error Loading Outlier Analysis</h4>
        <p>Unable to analyze outliers. Please try again.</p>
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="outlier-empty">
        <h4>No Analysis Data</h4>
        <p>Unable to load outlier analysis.</p>
      </div>
    );
  }

  return (
    <div className="outlier-container">
      {/* Tab Navigation */}
      <div className="outlier-tabs">
        <nav className="outlier-tab-nav" aria-label="Tabs">
          {[
            { key: 'analysis', label: 'Analysis' },
            { key: 'detection', label: 'Detection' },
            { key: 'treatment', label: 'Treatment' },
            { key: 'results', label: 'Results' }
          ].map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`outlier-tab-btn${activeTab === tab.key ? ' active' : ''}`}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        <div className="outlier-tab-content">
          {/* Analysis Tab */}
          {activeTab === 'analysis' && (
            <div>
              <div className="outlier-summary-grid">
                <div className="outlier-summary-box">
                  <div className="outlier-summary-value">
                    {analysis.summary?.total_consensus_outliers || 0}
                  </div>
                  <div className="outlier-summary-title">Total Outliers</div>
                </div>
                <div className="outlier-summary-box">
                  <div className="outlier-summary-value">
                    {analysis.summary?.columns_with_outliers || 0}
                  </div>
                  <div className="outlier-summary-title">Columns Affected</div>
                </div>
                <div className="outlier-summary-box">
                  <div className="outlier-summary-value">
                    {analysis.summary?.consensus_outlier_percentage?.toFixed(1) || 0}%
                  </div>
                  <div className="outlier-summary-title">Outlier Percentage</div>
                </div>
              </div>
              <div className="outlier-column-analysis">
                <h4 className="outlier-summary-title">Column-wise Outlier Analysis</h4>
                {Object.entries(analysis.column_analysis || {}).map(([column, colData]: [string, any]) => (
                  <div key={column} className="outlier-column-box">
                    <div className="outlier-column-header">
                      <div>
                        <h5 className="outlier-column-title">{column}</h5>
                        <p>
                          {colData.consensus_outliers?.length || 0} consensus outliers detected
                        </p>
                      </div>
                      <div className="outlier-column-percentage">
                        {((colData.consensus_outliers?.length || 0) / colData.total_values * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="outlier-methods-grid">
                      {Object.entries(colData.methods_results || {}).map(([method, result]: [string, any]) => (
                        <div key={method} className="outlier-method-box">
                          <div className="outlier-method-label">{method.replace('_', ' ')}</div>
                          <div className="outlier-method-value">
                            {result.outlier_count} ({result.outlier_percentage?.toFixed(1)}%)
                          </div>
                        </div>
                      ))}
                    </div>
                    {colData.recommended_treatment?.length > 0 && (
                      <div className="outlier-recommend-section">
                        <h6 className="outlier-recommend-title">Recommended Treatments:</h6>
                        <ul className="outlier-recommend-list">
                          {colData.recommended_treatment.slice(0, 2).map((rec: any, idx: number) => (
                            <li key={idx} className="outlier-recommend-item">
                              <span>{rec.method}</span>
                              <span> - {rec.reason}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                ))}
              </div>
              {analysis.recommendations?.length > 0 && (
                <div className="outlier-global-recommend">
                  <h4 className="outlier-recommend-title">Global Recommendations</h4>
                  <ul className="outlier-recommend-list">
                    {analysis.recommendations.map((rec: string, idx: number) => (
                      <li key={idx} className="outlier-recommend-item">{rec}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Detection Tab */}
          {activeTab === 'detection' && (
            <div className="outlier-detection-section">
              <div className="outlier-form-grid">
                <div>
                  <label className="outlier-form-label">Detection Method</label>
                  <select
                    value={selectedMethod}
                    onChange={(e) => setSelectedMethod(e.target.value)}
                    className="outlier-form-select"
                  >
                    {detectionMethods.map((method) => (
                      <option key={method.value} value={method.value}>
                        {method.label} - {method.description}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="outlier-form-label">Column (Optional)</label>
                  <select
                    value={selectedColumn}
                    onChange={(e) => setSelectedColumn(e.target.value)}
                    className="outlier-form-select"
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
                className="outlier-btn"
              >
                {detectionMutation.isPending ? 'Detecting...' : 'Detect Outliers'}
              </button>

              {/* Detection Results */}
              {detectionMutation.data && (
                <div className="outlier-preview-box">
                  <h4 className="outlier-summary-title">Detection Results</h4>
                  <div>
                    {Object.entries(detectionMutation.data.detection_result.outliers_detected || {}).map(([col, count]: [string, any]) => (
                      <div key={col} className="outlier-method-box" style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                        <span className="outlier-method-label">{col}</span>
                        <span className="outlier-method-value" style={{color: '#dc2626'}}>{count} outliers</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Treatment Tab */}
          {activeTab === 'treatment' && (
            <div className="outlier-treatment-section">
              <div className="outlier-form-grid">
                <div>
                  <label className="outlier-form-label">Treatment Method</label>
                  <select
                    value={treatmentMethod}
                    onChange={(e) => setTreatmentMethod(e.target.value)}
                    className="outlier-form-select"
                  >
                    {treatmentMethods.map((method) => (
                      <option key={method.value} value={method.value}>
                        {method.label} - {method.description}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="outlier-form-label">Column (Optional)</label>
                  <select
                    value={selectedColumn}
                    onChange={(e) => setSelectedColumn(e.target.value)}
                    className="outlier-form-select"
                  >
                    <option value="">All numeric columns</option>
                    {numericColumns.map((col) => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Treatment Options */}
              <div className="outlier-preview-box">
                <h4 className="outlier-summary-title">Treatment Options</h4>
                <div className="outlier-form-grid">
                  <div>
                    <label className="outlier-form-label">Detection Method</label>
                    <select
                      value={treatmentOptions.detection_method}
                      onChange={(e) => setTreatmentOptions({...treatmentOptions, detection_method: e.target.value})}
                      className="outlier-form-select"
                    >
                      <option value="iqr">IQR</option>
                      <option value="z_score">Z-Score</option>
                      <option value="modified_z_score">Modified Z-Score</option>
                    </select>
                  </div>
                  <div>
                    <label className="outlier-form-label">Lower Percentile</label>
                    <input
                      type="number"
                      min="0"
                      max="50"
                      step="0.1"
                      value={treatmentOptions.lower_percentile}
                      onChange={(e) => setTreatmentOptions({...treatmentOptions, lower_percentile: parseFloat(e.target.value)})}
                      className="outlier-form-input"
                    />
                  </div>
                  <div>
                    <label className="outlier-form-label">Upper Percentile</label>
                    <input
                      type="number"
                      min="50"
                      max="100"
                      step="0.1"
                      value={treatmentOptions.upper_percentile}
                      onChange={(e) => setTreatmentOptions({...treatmentOptions, upper_percentile: parseFloat(e.target.value)})}
                      className="outlier-form-input"
                    />
                  </div>
                </div>
              </div>
              <div style={{display: 'flex', gap: '1rem', marginTop: '1rem'}}>
                <button
                  onClick={() => previewMutation.mutate({...treatmentOptions, column: selectedColumn || undefined})}
                  disabled={previewMutation.isPending}
                  className="outlier-btn outlier-btn-yellow"
                >
                  {previewMutation.isPending ? 'Previewing...' : 'Preview Treatment'}
                </button>
                <button
                  onClick={() => treatmentMutation.mutate({...treatmentOptions, column: selectedColumn || undefined})}
                  disabled={treatmentMutation.isPending}
                  className="outlier-btn outlier-btn-green"
                >
                  {treatmentMutation.isPending ? 'Applying...' : 'Apply Treatment'}
                </button>
              </div>

              {/* Preview Results */}
              {previewMutation.data && (
                <div className="outlier-preview-box">
                  <h4 className="outlier-summary-title">Treatment Preview</h4>
                  <div>
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
            <div className="outlier-results-section">
              {treatmentMutation.data ? (
                <div className="outlier-results-box">
                  <div className="outlier-results-grid">
                    <div>
                      <div className="outlier-results-value">
                        {treatmentMutation.data.treatment_info.original_shape[0]}
                      </div>
                      <div className="outlier-summary-title">Original Rows</div>
                    </div>
                    <div>
                      <div className="outlier-results-value">
                        {treatmentMutation.data.treatment_info.final_shape[0]}
                      </div>
                      <div className="outlier-summary-title">Final Rows</div>
                    </div>
                    <div>
                      <div className="outlier-results-value">
                        {treatmentMutation.data.treatment_info.outliers_removed || treatmentMutation.data.treatment_info.outliers_treated || 0}
                      </div>
                      <div className="outlier-summary-title">Outliers Processed</div>
                    </div>
                    <div>
                      <div className="outlier-results-value">
                        {treatmentMutation.data.treatment_info.method}
                      </div>
                      <div className="outlier-summary-title">Method Used</div>
                    </div>
                  </div>
                  <button
                    onClick={() => {
                      const treatedId = treatmentMutation.data.treated_dataset_id;
                      window.open(`${API_BASE_URL}/api/v1/ai-insights/${treatedId}/download/csv`, '_blank');
                    }}
                    className="outlier-download-btn"
                  >
                    Download Treated Dataset (No Outliers)
                  </button>
                </div>
              ) : (
                <div className="outlier-empty">
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
