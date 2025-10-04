import React from 'react';
import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import EDALayout from '../components/eda/EDALayout';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface EDAResults {
  dataset_id: string;
  analyses: {
    basic_statistics: any;
    missing_values: any;
    correlations: any;
    distributions: any;
    outliers: any;
  };
  visualizations: any[];
  summary: string | {
    total_analyses?: number;
    completed_at?: string;
    status?: string;
  };
  narrative?: string;
}

const EDA: React.FC = () => {
  const { datasetId } = useParams<{ datasetId: string }>();

  // Fetch EDA results
  const { data: edaResults, isLoading, error } = useQuery<EDAResults>({
    queryKey: ['eda', datasetId],
    queryFn: async () => {
      if (!datasetId) throw new Error('Dataset ID is required');
      const response = await axios.get(`${API_BASE_URL}/api/v1/eda/${datasetId}/results`);
      return response.data;
    },
    enabled: !!datasetId,
  });

  // Fetch visualizations
  const { data: visualizations } = useQuery({
    queryKey: ['visualizations', datasetId],
    queryFn: async () => {
      if (!datasetId) throw new Error('Dataset ID is required');
      const response = await axios.get(`${API_BASE_URL}/api/v1/eda/${datasetId}/visualizations`);
      return response.data;
    },
    enabled: !!datasetId,
  });

  // Fetch statistical analysis
  const { data: statisticalAnalysis } = useQuery({
    queryKey: ['statistics', datasetId],
    queryFn: async () => {
      if (!datasetId) throw new Error('Dataset ID is required');
      const response = await axios.get(`${API_BASE_URL}/api/v1/eda/${datasetId}/statistics`);
      return response.data;
    },
    enabled: !!datasetId,
  });

  if (isLoading) {
    return (
      <div className="card">
        <div className="text-center py-8">
          <div className="loading mb-4"></div>
          <p>Loading EDA results...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <div className="text-center py-8 text-red-600">
          <p>Error loading EDA results. Please try again.</p>
        </div>
      </div>
    );
  }

  if (!edaResults) {
    return (
      <div className="card">
        <div className="text-center py-8">
          <p>No EDA results found for this dataset.</p>
        </div>
      </div>
    );
  }

  return (
    <div>
      <EDALayout 
        datasetId={datasetId!}
        edaResults={edaResults}
        statisticalAnalysis={statisticalAnalysis}
        visualizations={visualizations}
      />
      
      {/* AI Narrative */}
      {edaResults.narrative && (
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">🤖 AI Analysis Summary</h2>
          </div>
          <div className="prose max-w-none">
            <div className="whitespace-pre-wrap text-gray-700">
              {edaResults.narrative}
            </div>
          </div>
        </div>
      )}

      {/* Summary */}
      {edaResults.summary && typeof edaResults.summary === 'string' && (
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Summary</h2>
          </div>
          <div className="text-gray-700">
            {edaResults.summary}
          </div>
        </div>
      )}
    </div>
  );
};

export default EDA;
