// frontend/src/services/aiService.ts - FIXED VERSION

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
  // MISSING VALUES - FIXED METHODS
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
   * Apply missing value treatment - FIXED
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
      
      if (options.column) {
        params.append('column', options.column);
      }
      if (options.n_neighbors !== undefined) {
        params.append('n_neighbors', options.n_neighbors.toString());
      }
      if (options.constant_value !== undefined) {
        params.append('constant_value', options.constant_value);
      }
      if (options.threshold !== undefined) {
        params.append('threshold', options.threshold.toString());
      }

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
    
    if (options.column) {
      params.append('column', options.column);
    }
    if (options.n_neighbors !== undefined) {
      params.append('n_neighbors', options.n_neighbors.toString());
    }
    if (options.constant_value !== undefined) {
      params.append('constant_value', options.constant_value);
    }
    if (options.threshold !== undefined) {
      params.append('threshold', options.threshold.toString());
    }

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
  // OTHER AI INSIGHTS METHODS
  // ============================================================================

  /**
   * Get correlation insights
   */
  getCorrelationInsights: async (datasetId: string) => {
    const response = await apiClient.get(`/ai-insights/${datasetId}/correlations`);
    return response.data;
  },

  /**
   * Get outlier insights
   */
  getOutlierInsights: async (datasetId: string) => {
    const response = await apiClient.get(`/ai-insights/${datasetId}/outliers`);
    return response.data;
  },

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

  /**
   * Get distribution insights
   */
  getDistributionInsights: async (datasetId: string) => {
    const response = await apiClient.get(`/ai-insights/${datasetId}/distributions`);
    return response.data;
  },

  /**
   * Get data quality insights
   */
  getDataQualityInsights: async (datasetId: string) => {
    const response = await apiClient.get(`/ai-insights/${datasetId}/data-quality`);
    return response.data;
  },

  /**
   * Get feature importance insights
   */
  getFeatureImportanceInsights: async (datasetId: string) => {
    const response = await apiClient.get(`/ai-insights/${datasetId}/features`);
    return response.data;
  },
};

export default aiService;