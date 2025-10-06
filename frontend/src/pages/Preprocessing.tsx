import React, { useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

type SourceOption = 'auto' | 'original' | 'outlier_treated';

interface StepResponse {
  dataset_id: string;
  shape_before: number[];
  shape_after: number[];
  steps?: any;
  encoding?: any;
  scaling?: any;
  reduction?: any;
}

const Preprocessing: React.FC = () => {
  const { datasetId } = useParams<{ datasetId: string }>();
  const [source, setSource] = useState<SourceOption>('auto');
  const [useGemini, setUseGemini] = useState<boolean | undefined>(undefined);

  const queryParams = useMemo(() => {
    const params = new URLSearchParams();
    params.set('source', source);
    if (useGemini !== undefined) params.set('use_gemini', String(useGemini));
    return params.toString();
  }, [source, useGemini]);

  const { data: preprocessResult, isLoading: loadingPreprocess } = useQuery<StepResponse>({
    queryKey: ['preprocess', datasetId, source, useGemini],
    queryFn: async () => {
      if (!datasetId) throw new Error('Dataset ID is required');
      const res = await axios.post(`${API_BASE_URL}/api/v1/preprocessing/${datasetId}/preprocess?${queryParams}`);
      return res.data;
    },
    enabled: !!datasetId,
  });

  const { data: encodeResult } = useQuery<StepResponse>({
    queryKey: ['encode', datasetId, source, useGemini],
    queryFn: async () => {
      if (!datasetId) throw new Error('Dataset ID is required');
      const res = await axios.post(`${API_BASE_URL}/api/v1/preprocessing/${datasetId}/encode?${queryParams}`);
      return res.data;
    },
    enabled: !!datasetId,
  });

  const { data: scaleResult } = useQuery<StepResponse>({
    queryKey: ['scale', datasetId, source, useGemini],
    queryFn: async () => {
      if (!datasetId) throw new Error('Dataset ID is required');
      const res = await axios.post(`${API_BASE_URL}/api/v1/preprocessing/${datasetId}/scale?${queryParams}`);
      return res.data;
    },
    enabled: !!datasetId,
  });

  const { data: reduceResult } = useQuery<StepResponse>({
    queryKey: ['reduce', datasetId, source, useGemini],
    queryFn: async () => {
      if (!datasetId) throw new Error('Dataset ID is required');
      const res = await axios.post(`${API_BASE_URL}/api/v1/preprocessing/${datasetId}/reduce?${queryParams}`);
      return res.data;
    },
    enabled: !!datasetId,
  });

  return (
    <div>
      <div className="mb-4 flex gap-2 justify-end">
        <button
          className="btn btn-primary"
          onClick={async () => {
            if (!datasetId) return;
            await axios.post(`${API_BASE_URL}/api/v1/preprocessing/${datasetId}/preprocess/save?${queryParams}`);
          }}
        >
          Run & Save Preprocessed CSV
        </button>
        <a
          className="btn"
          href={`${API_BASE_URL}/api/v1/preprocessing/${datasetId}/download`}
          target="_blank"
          rel="noreferrer"
        >
          Download Preprocessed CSV
        </a>
      </div>
      <div className="card mb-6">
        <div className="card-header">
          <h2 className="card-title">Preprocessing Controls</h2>
        </div>
        <div className="flex flex-wrap gap-4 items-center">
          <div>
            <label className="block text-sm font-medium mb-1">Dataset Source</label>
            <select
              className="select select-bordered"
              value={source}
              onChange={(e) => setSource(e.target.value as SourceOption)}
            >
              <option value="auto">Auto (prefer outlier_treated)</option>
              <option value="original">Original</option>
              <option value="outlier_treated">Outlier Treated</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Use Gemini</label>
            <select
              className="select select-bordered"
              value={useGemini === undefined ? 'auto' : useGemini ? 'true' : 'false'}
              onChange={(e) => {
                const val = e.target.value;
                setUseGemini(val === 'auto' ? undefined : val === 'true');
              }}
            >
              <option value="auto">Auto</option>
              <option value="true">Force On</option>
              <option value="false">Force Off</option>
            </select>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6">
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Full Preprocess</h2>
          </div>
          {loadingPreprocess ? (
            <div className="p-4">Processing...</div>
          ) : (
            <pre className="p-4 overflow-auto bg-gray-50 rounded">
{JSON.stringify(preprocessResult, null, 2)}
            </pre>
          )}
        </div>

        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Encoding</h2>
          </div>
          <pre className="p-4 overflow-auto bg-gray-50 rounded">
{JSON.stringify(encodeResult, null, 2)}
          </pre>
        </div>

        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Scaling</h2>
          </div>
          <pre className="p-4 overflow-auto bg-gray-50 rounded">
{JSON.stringify(scaleResult, null, 2)}
          </pre>
        </div>

        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Dimensionality Reduction</h2>
          </div>
          <pre className="p-4 overflow-auto bg-gray-50 rounded">
{JSON.stringify(reduceResult, null, 2)}
          </pre>
        </div>
      </div>
    </div>
  );
};

export default Preprocessing;


