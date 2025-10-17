// frontend/src/services/aiService.ts - CLEANED & FIXED VERSION

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  headers: {
    'Content-Type': 'application/json',
  },
});

const aiService = {
  // ============================================================================
  // MISSING VALUES METHODS
  // ============================================================================

  /**
   * Get missing values analysis for a dataset
   */
  getMissingValuesInsights: async (datasetId: string) => {
    console.log('📊 Fetching missing values insights for:', datasetId);
    const response = await apiClient.get(`/ai-insights/${datasetId}/missing-values`);
    console.log('✅ Received analysis:', response.data);
    return response.data;
  },

  /**
   * Apply missing value treatment
   */
  treatMissingValues: async (
    datasetId: string,
    method: string,
    options: {
      column?: string;
      n_neighbors?: number;
      constant_value?: string;
      threshold?: number;
    } = {}
  ) => {
    console.log('🔧 Applying treatment:', { datasetId, method, options });

    try {
      const params = new URLSearchParams();
      params.append('method', method);

      if (options.column) params.append('column', options.column);
      if (options.n_neighbors !== undefined)
        params.append('n_neighbors', options.n_neighbors.toString());
      if (options.constant_value !== undefined)
        params.append('constant_value', options.constant_value);
      if (options.threshold !== undefined)
        params.append('threshold', options.threshold.toString());

      const response = await apiClient.post(
        `/ai-insights/${datasetId}/missing-values/treat?${params.toString()}`
      );

      console.log('✅ Treatment response:', response.data);
      return response.data;
    } catch (error: any) {
      console.error('❌ Treatment error:', error);
      console.error('Error details:', error.response?.data);
      throw error;
    }
  },

  /**
   * Preview missing value treatment effects
   */
  previewMissingValuesTreatment: async (
    datasetId: string,
    method: string,
    options: {
      column?: string;
      n_neighbors?: number;
      constant_value?: string;
      threshold?: number;
    } = {}
  ) => {
    console.log('👁️ Previewing treatment:', { datasetId, method, options });

    const params = new URLSearchParams();
    params.append('method', method);

    if (options.column) params.append('column', options.column);
    if (options.n_neighbors !== undefined)
      params.append('n_neighbors', options.n_neighbors.toString());
    if (options.constant_value !== undefined)
      params.append('constant_value', options.constant_value);
    if (options.threshold !== undefined)
      params.append('threshold', options.threshold.toString());

    const response = await apiClient.get(
      `/ai-insights/${datasetId}/missing-values/preview?${params.toString()}`
    );

    console.log('✅ Preview response:', response.data);
    return response.data;
  },

  /**
   * Get treatment status for a dataset
   */
  getTreatmentStatus: async (datasetId: string) => {
    console.log('🔍 Checking treatment status for:', datasetId);
    const response = await apiClient.get(`/ai-insights/${datasetId}/treatment-status`);
    console.log('✅ Status:', response.data);
    return response.data;
  },

  /**
   * Download original dataset as CSV
   */
  downloadOriginalDataset: async (datasetId: string) => {
    console.log('📥 Downloading original dataset:', datasetId);
    const response = await apiClient.get(`/ai-insights/${datasetId}/download/csv`, {
      responseType: 'blob',
    });
    console.log('✅ Download complete');
    return response.data;
  },

  /**
   * Download treated dataset as CSV
   */
  downloadTreatedDataset: async (datasetId: string) => {
    console.log('📥 Downloading treated dataset:', datasetId);
    const treatedId = `${datasetId}_treated`;
    const response = await apiClient.get(`/ai-insights/${treatedId}/download/csv`, {
      responseType: 'blob',
    });
    console.log('✅ Download complete');
    return response.data;
  },

  // ============================================================================
  // OUTLIER METHODS
  // ============================================================================

  /**
   * Get comprehensive outlier analysis
   */
  getOutlierAnalysis: async (datasetId: string) => {
    console.log('📊 Fetching outlier analysis for:', datasetId);
    const response = await apiClient.get(`/ai-insights/${datasetId}/outliers/analysis`);
    console.log('✅ Received analysis:', response.data);
    return response.data;
  },

  /**
   * Detect outliers using specified method
   */
  detectOutliers: async (
    datasetId: string,
    options: {
      method?: string;
      column?: string;
      threshold?: number;
      multiplier?: number;
      lower_percentile?: number;
      upper_percentile?: number;
    } = {}
  ) => {
    console.log('🔍 Detecting outliers:', { datasetId, options });

    const params = new URLSearchParams();
    params.append('method', options.method || 'z_score');

    if (options.column) params.append('column', options.column);
    if (options.threshold !== undefined)
      params.append('threshold', options.threshold.toString());
    if (options.multiplier !== undefined)
      params.append('multiplier', options.multiplier.toString());
    if (options.lower_percentile !== undefined)
      params.append('lower_percentile', options.lower_percentile.toString());
    if (options.upper_percentile !== undefined)
      params.append('upper_percentile', options.upper_percentile.toString());

    const response = await apiClient.get(
      `/ai-insights/${datasetId}/outliers/detect?${params.toString()}`
    );

    console.log('✅ Detection response:', response.data);
    return response.data;
  },

  /**
   * Get outlier summary
   */
  getOutlierSummary: async (datasetId: string) => {
    console.log('📈 Fetching outlier summary for:', datasetId);
    const response = await apiClient.get(`/ai-insights/${datasetId}/outliers/summary`);
    console.log('✅ Received summary:', response.data);
    return response.data;
  },

  /**
   * Preview outlier treatment
   */
  previewOutlierTreatment: async (
    datasetId: string,
    method: string,
    options: {
      column?: string;
      detection_method?: string;
      lower_percentile?: number;
      upper_percentile?: number;
      z_threshold?: number;
    } = {}
  ) => {
    console.log('👁️ Previewing treatment:', { datasetId, method, options });

    const params = new URLSearchParams();
    params.append('method', method);

    if (options.column) params.append('column', options.column);
    params.append('detection_method', options.detection_method || 'iqr');

    if (options.lower_percentile !== undefined)
      params.append('lower_percentile', options.lower_percentile.toString());
    if (options.upper_percentile !== undefined)
      params.append('upper_percentile', options.upper_percentile.toString());
    if (options.z_threshold !== undefined)
      params.append('z_threshold', options.z_threshold.toString());

    const response = await apiClient.get(
      `/ai-insights/${datasetId}/outliers/preview-treatment?${params.toString()}`
    );

    console.log('✅ Preview response:', response.data);
    return response.data;
  },

  /**
   * Apply outlier treatment
   */
  treatOutliers: async (
    datasetId: string,
    method: string,
    options: {
      column?: string;
      detection_method?: string;
      lower_percentile?: number;
      upper_percentile?: number;
      z_threshold?: number;
    } = {}
  ) => {
    console.log('🔧 Applying treatment:', { datasetId, method, options });

    try {
      const params = new URLSearchParams();
      params.append('method', method);

      if (options.column) params.append('column', options.column);
      params.append('detection_method', options.detection_method || 'iqr');

      if (options.lower_percentile !== undefined)
        params.append('lower_percentile', options.lower_percentile.toString());
      if (options.upper_percentile !== undefined)
        params.append('upper_percentile', options.upper_percentile.toString());
      if (options.z_threshold !== undefined)
        params.append('z_threshold', options.z_threshold.toString());

      const response = await apiClient.post(
        `/ai-insights/${datasetId}/outliers/treat?${params.toString()}`
      );

      console.log('✅ Treatment response:', response.data);
      return response.data;
    } catch (error: any) {
      console.error('❌ Treatment error:', error);
      console.error('Error details:', error.response?.data);
      throw error;
    }
  },

  /**
   * Get outlier treatment status
   */
  getOutlierTreatmentStatus: async (datasetId: string) => {
    console.log('🔍 Checking outlier treatment status for:', datasetId);
    const response = await apiClient.get(`/ai-insights/${datasetId}/outliers/treatment-status`);
    console.log('✅ Status:', response.data);
    return response.data;
  },

  /**
   * Get outlier treatment history
   */
  getOutlierTreatmentHistory: async (datasetId: string) => {
    console.log('📜 Fetching treatment history for:', datasetId);
    const response = await apiClient.get(`/ai-insights/${datasetId}/outliers/treatment-history`);
    console.log('✅ History:', response.data);
    return response.data;
  },

  /**
   * Download treated dataset
   */
  downloadOutlierTreatedDataset: async (datasetId: string) => {
    console.log('📥 Downloading outlier-treated dataset:', datasetId);
    const treatedId = `${datasetId}_outliers_treated`;
    const response = await apiClient.get(`/ai-insights/${treatedId}/download/csv`, {
      responseType: 'blob',
    });
    console.log('✅ Download complete');
    return response.data;
  },

  // ============================================================================
  // OTHER AI INSIGHTS
  // ============================================================================

  /**
   * Get dataset overview
   */
  getDatasetOverview: async (datasetId: string) => {
    const response = await apiClient.get(`/ai-insights/${datasetId}/overview`);
    return response.data;
  },

  /**
   * Get statistical insights
   */
  getStatisticalInsights: async (datasetId: string) => {
    const response = await apiClient.get(`/ai-insights/${datasetId}/statistics`);
    return response.data;
  },

  getCorrelationInsights: async (datasetId: string) => {
    const response = await apiClient.get(`/ai-insights/${datasetId}/correlations`);
    return response.data;
  },

  getDistributionInsights: async (datasetId: string) => {
    const response = await apiClient.get(`/ai-insights/${datasetId}/distributions`);
    return response.data;
  },

  getDataQualityInsights: async (datasetId: string) => {
    const response = await apiClient.get(`/ai-insights/${datasetId}/data-quality`);
    return response.data;
  },

  getFeatureImportanceInsights: async (datasetId: string) => {
    const response = await apiClient.get(`/ai-insights/${datasetId}/features`);
    return response.data;
  },
};

export default aiService;
