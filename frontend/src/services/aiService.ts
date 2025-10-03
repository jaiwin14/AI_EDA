/**
 * AI Insights Service
 * Handles all AI-powered analysis requests to the backend
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
      throw new Error(`Failed to get correlation insights: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Get enhanced correlation analysis with AI insights
   */
  async getEnhancedCorrelationAnalysis(datasetId: string): Promise<any> {
    const response = await fetch(`${this.baseURL}/ai-insights/${datasetId}/correlations`);
    if (!response.ok) {
      throw new Error(`Failed to get enhanced correlation analysis: ${response.statusText}`);
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
   * Treat missing values using specified method
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
    const response = await fetch(`${this.baseURL}/ai-insights/${datasetId}/missing-values/treat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        method,
        ...options,
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to treat missing values: ${response.statusText}`);
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
    const params = new URLSearchParams({
      method,
      ...Object.fromEntries(
        Object.entries(options).map(([key, value]) => [key, String(value)])
      ),
    });

    const response = await fetch(`${this.baseURL}/ai-insights/${datasetId}/missing-values/preview?${params}`);
    if (!response.ok) {
      throw new Error(`Failed to preview missing values treatment: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Download treated dataset as CSV
   */
  async downloadTreatedDataset(datasetId: string): Promise<Blob> {
    const response = await fetch(`${this.baseURL}/ai-insights/${datasetId}/download/csv`);
    if (!response.ok) {
      throw new Error(`Failed to download dataset: ${response.statusText}`);
    }
    return response.blob();
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
      headers: {
        'Content-Type': 'application/json',
      },
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
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        method,
        ...options,
      }),
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
      headers: {
        'Content-Type': 'application/json',
      },
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
      headers: {
        'Content-Type': 'application/json',
      },
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
