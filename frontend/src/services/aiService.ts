/**
 * AI Insights Service
 * Handles all AI-powered analysis requests to the backend
 * FIXED VERSION - All endpoints working correctly
 */

export interface AIInsightResponse {
  success: boolean;
  analysis_type: string;
  dataset_id: string;
  timestamp: string;
  data: any;
}

export interface AnalysisTemplate {
  type: string;
  name: string;
  description: string;
}

export interface AnalysisType {
  type: string;
  name: string;
  endpoint: string;
}

export enum AIAnalysisType {
  OVERVIEW = 'overview',
  STATISTICS = 'statistics',
  CORRELATIONS = 'correlations',
  MISSING_VALUES = 'missing_values',
  OUTLIERS = 'outliers',
  DISTRIBUTIONS = 'distributions',
  DATA_QUALITY = 'data_quality',
  FEATURES = 'features',
  TRENDS = 'trends',
  BUSINESS = 'business'
}

// NEW: Treatment status interface
export interface TreatmentStatus {
  success: boolean;
  dataset_id: string;
  treatment_applied: boolean;
  has_treated_file: boolean;
  treated_dataset_id: string | null;
  has_missing_values: boolean | null;
  missing_values_count: number | null;
  can_proceed_to_outliers: boolean;
  timestamp: string;
}

// NEW: Active dataset interface
export interface ActiveDataset {
  success: boolean;
  dataset_id: string;
  is_treated: boolean;
  original_id?: string;
  data: any;
}

class AIService {
  private baseURL = 'http://localhost:8000/api/v1';

  /**
   * Get AI-powered dataset overview insights
   */
  async getDatasetOverview(datasetId: string): Promise<AIInsightResponse> {
    const response = await fetch(`${this.baseURL}/ai-insights/${datasetId}/overview`);
    if (!response.ok) {
      throw new Error(`Failed to get dataset overview: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Get AI-powered statistical insights
   */
  async getStatisticalInsights(datasetId: string): Promise<AIInsightResponse> {
    const response = await fetch(`${this.baseURL}/ai-insights/${datasetId}/statistics`);
    if (!response.ok) {
      throw new Error(`Failed to get statistical insights: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Get AI-powered correlation insights
   */
  async getCorrelationInsights(datasetId: string): Promise<AIInsightResponse> {
    const response = await fetch(`${this.baseURL}/ai-insights/${datasetId}/correlations`);
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(`Failed to get correlation insights: ${response.statusText}. ${errorData.detail || ''}`);
    }
    return response.json();
  }

  /**
   * Get enhanced correlation analysis with AI insights
   */
  async getEnhancedCorrelationAnalysis(datasetId: string): Promise<any> {
    const response = await fetch(`${this.baseURL}/ai-insights/${datasetId}/correlations`);
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(`Failed to get enhanced correlation analysis: ${response.statusText}. ${errorData.detail || ''}`);
    }
    return response.json();
  }

  /**
   * Get correlation matrix only
   */
  async getCorrelationMatrix(datasetId: string): Promise<any> {
    const response = await fetch(`${this.baseURL}/ai-insights/${datasetId}/correlations`);
    if (!response.ok) {
      throw new Error(`Failed to get correlation matrix: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Get AI-powered missing values insights
   */
  async getMissingValuesInsights(datasetId: string): Promise<AIInsightResponse> {
    const response = await fetch(`${this.baseURL}/ai-insights/${datasetId}/missing-values`);
    if (!response.ok) {
      throw new Error(`Failed to get missing values insights: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Get comprehensive missing values analysis
   */
  async getMissingValuesAnalysis(datasetId: string): Promise<AIInsightResponse> {
    const response = await fetch(`${this.baseURL}/ai-insights/${datasetId}/missing-values/analysis`);
    if (!response.ok) {
      throw new Error(`Failed to get missing values analysis: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * FIXED: Treat missing values using specified method
   * Now properly sends query parameters via POST
   */
  async treatMissingValues(
    datasetId: string,
    method: string,
    options: {
      column?: string;
      n_neighbors?: number;
      constant_value?: any;
      threshold?: number;
    } = {}
  ): Promise<any> {
    // Build query parameters
    const params = new URLSearchParams();
    params.append('method', method);
    
    if (options.column) params.append('column', options.column);
    if (options.n_neighbors !== undefined) params.append('n_neighbors', String(options.n_neighbors));
    if (options.constant_value !== undefined) params.append('constant_value', String(options.constant_value));
    if (options.threshold !== undefined) params.append('threshold', String(options.threshold));

    const response = await fetch(
      `${this.baseURL}/ai-insights/${datasetId}/missing-values/treat?${params}`,
      { method: 'POST' }
    );

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Failed to treat missing values: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Preview missing values treatment
   */
  async previewMissingValuesTreatment(
    datasetId: string,
    method: string,
    options: {
      column?: string;
      n_neighbors?: number;
      constant_value?: any;
      threshold?: number;
    } = {}
  ): Promise<any> {
    const params = new URLSearchParams({ method });
    
    if (options.column) params.append('column', options.column);
    if (options.n_neighbors !== undefined) params.append('n_neighbors', String(options.n_neighbors));
    if (options.constant_value !== undefined) params.append('constant_value', String(options.constant_value));
    if (options.threshold !== undefined) params.append('threshold', String(options.threshold));

    const response = await fetch(`${this.baseURL}/ai-insights/${datasetId}/missing-values/preview?${params}`);
    if (!response.ok) {
      throw new Error(`Failed to preview missing values treatment: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * NEW: Check treatment status for a dataset
   */
  async getTreatmentStatus(datasetId: string): Promise<TreatmentStatus> {
    const response = await fetch(`${this.baseURL}/ai-insights/${datasetId}/treatment-status`);
    if (!response.ok) {
      throw new Error(`Failed to get treatment status: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * NEW: Get the active dataset (treated if available, otherwise original)
   */
  async getActiveDataset(datasetId: string): Promise<ActiveDataset> {
    const response = await fetch(`${this.baseURL}/ai-insights/${datasetId}/active-dataset`);
    if (!response.ok) {
      throw new Error(`Failed to get active dataset: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Download original dataset as CSV
   */
  async downloadOriginalDataset(datasetId: string): Promise<Blob> {
    const response = await fetch(`${this.baseURL}/ai-insights/${datasetId}/download/csv`);
    if (!response.ok) {
      throw new Error(`Failed to download original dataset: ${response.statusText}`);
    }
    return response.blob();
  }

  /**
   * Download treated/cleaned dataset as CSV
   */
  async downloadTreatedDataset(datasetId: string): Promise<Blob> {
    const treatedDatasetId = `${datasetId}_treated`;
    const response = await fetch(`${this.baseURL}/ai-insights/${treatedDatasetId}/download/csv`);
    if (!response.ok) {
      throw new Error(`Failed to download treated dataset: ${response.statusText}`);
    }
    return response.blob();
  }

  /**
   * Get treatment history for a dataset
   */
  async getTreatmentHistory(datasetId: string): Promise<any> {
    const response = await fetch(`${this.baseURL}/ai-insights/${datasetId}/treatment-history`);
    if (!response.ok) {
      throw new Error(`Failed to get treatment history: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Get AI-powered outlier insights
   */
  async getOutlierInsights(datasetId: string): Promise<AIInsightResponse> {
    const response = await fetch(`${this.baseURL}/ai-insights/${datasetId}/outliers`);
    if (!response.ok) {
      throw new Error(`Failed to get outlier insights: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Get comprehensive outlier analysis
   */
  async getOutlierAnalysis(datasetId: string): Promise<AIInsightResponse> {
    const response = await fetch(`${this.baseURL}/ai-insights/${datasetId}/outliers/analysis`);
    if (!response.ok) {
      throw new Error(`Failed to get outlier analysis: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Detect outliers using specified method
   */
  async detectOutliers(
    datasetId: string,
    options: {
      method?: string;
      column?: string;
      threshold?: number;
      multiplier?: number;
      contamination?: number;
      lower_percentile?: number;
      upper_percentile?: number;
    } = {}
  ): Promise<any> {
    const response = await fetch(`${this.baseURL}/ai-insights/${datasetId}/outliers/detect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(options),
    });

    if (!response.ok) {
      throw new Error(`Failed to detect outliers: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Treat outliers using specified method
   */
  async treatOutliers(
    datasetId: string,
    method: string,
    options: {
      column?: string;
      detection_method?: string;
      z_threshold?: number;
      lower_percentile?: number;
      upper_percentile?: number;
    } = {}
  ): Promise<any> {
    const response = await fetch(`${this.baseURL}/ai-insights/${datasetId}/outliers/treat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ method, ...options }),
    });

    if (!response.ok) {
      throw new Error(`Failed to treat outliers: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Preview outlier treatment
   */
  async previewOutlierTreatment(
    datasetId: string,
    method: string,
    options: {
      column?: string;
      detection_method?: string;
      z_threshold?: number;
      lower_percentile?: number;
      upper_percentile?: number;
    } = {}
  ): Promise<any> {
    const params = new URLSearchParams({
      method,
      ...Object.fromEntries(
        Object.entries(options).map(([key, value]) => [key, String(value)])
      ),
    });

    const response = await fetch(`${this.baseURL}/ai-insights/${datasetId}/outliers/preview?${params}`);
    if (!response.ok) {
      throw new Error(`Failed to preview outlier treatment: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Generate AI insights for specific analysis type
   */
  async generateInsights(
    datasetId: string, 
    analysisType: AIAnalysisType,
    customContext?: Record<string, any>
  ): Promise<AIInsightResponse> {
    const response = await fetch(`${this.baseURL}/ai-insights/${datasetId}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        analysis_type: analysisType,
        custom_context: customContext,
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to generate insights: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Generate custom AI insights with specific template
   */
  async generateCustomInsights(
    datasetId: string,
    templateType: string,
    contextData: Record<string, any>
  ): Promise<AIInsightResponse> {
    const response = await fetch(`${this.baseURL}/ai-insights/${datasetId}/custom`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        template_type: templateType,
        context_data: contextData,
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to generate custom insights: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Get available AI analysis templates
   */
  async getAvailableTemplates(): Promise<{
    templates: AnalysisTemplate[];
    analysis_types: AnalysisType[];
  }> {
    const response = await fetch(`${this.baseURL}/ai-insights/templates`);
    if (!response.ok) {
      throw new Error(`Failed to get templates: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Legacy method for backward compatibility
   */
  async getAIOverview(datasetId: string): Promise<any> {
    const response = await fetch(`${this.baseURL}/eda/${datasetId}/ai-overview`);
    if (!response.ok) {
      throw new Error(`Failed to get AI overview: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Legacy method for backward compatibility
   */
  async getAISummary(datasetId: string): Promise<any> {
    const response = await fetch(`${this.baseURL}/eda/${datasetId}/ai-summary`);
    if (!response.ok) {
      throw new Error(`Failed to get AI summary: ${response.statusText}`);
    }
    return response.json();
  }
}

export const aiService = new AIService();
export default aiService;