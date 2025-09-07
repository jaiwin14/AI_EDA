import React, { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface ModelResult {
  model_id: string;
  model_name: string;
  status: string;
  metrics: {
    train: any;
    test: any;
  };
  training_time: number;
  created_at: string;
}

interface TrainingStatus {
  status: string;
  progress: number;
  current_model: string;
  message: string;
}

const Models: React.FC = () => {
  const { datasetId } = useParams<{ datasetId: string }>();
  const navigate = useNavigate();
  const [selectedTargetColumn, setSelectedTargetColumn] = useState<string>('');
  const [taskType, setTaskType] = useState<string>('auto');

  // Fetch dataset info
  const { data: datasetInfo } = useQuery({
    queryKey: ['dataset-info', datasetId],
    queryFn: async () => {
      if (!datasetId) throw new Error('Dataset ID is required');
      const response = await axios.get(`${API_BASE_URL}/api/upload/datasets/${datasetId}`);
      return response.data;
    },
    enabled: !!datasetId,
  });

  // Fetch trained models
  const { data: models, refetch: refetchModels } = useQuery<ModelResult[]>({
    queryKey: ['models', datasetId],
    queryFn: async () => {
      if (!datasetId) throw new Error('Dataset ID is required');
      const response = await axios.get(`${API_BASE_URL}/api/models/${datasetId}`);
      return response.data;
    },
    enabled: !!datasetId,
  });

  // Fetch training status
  const { data: trainingStatus } = useQuery<TrainingStatus>({
    queryKey: ['training-status', datasetId],
    queryFn: async () => {
      if (!datasetId) throw new Error('Dataset ID is required');
      const response = await axios.get(`${API_BASE_URL}/api/models/${datasetId}/status`);
      return response.data;
    },
    enabled: !!datasetId,
    refetchInterval: trainingStatus?.status === 'training' ? 2000 : false,
  });

  // Start training mutation
  const trainModelsMutation = useMutation({
    mutationFn: async (params: { target_column: string; task_type: string }) => {
      if (!datasetId) throw new Error('Dataset ID is required');
      const response = await axios.post(`${API_BASE_URL}/api/models/${datasetId}/train`, params);
      return response.data;
    },
    onSuccess: () => {
      toast.success('Model training started!');
      refetchModels();
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Training failed');
    },
  });

  const handleStartTraining = () => {
    if (!selectedTargetColumn) {
      toast.error('Please select a target column');
      return;
    }
    trainModelsMutation.mutate({
      target_column: selectedTargetColumn,
      task_type: taskType,
    });
  };

  const formatMetric = (value: number) => {
    return typeof value === 'number' ? value.toFixed(4) : 'N/A';
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'status-success';
      case 'training': return 'status-warning';
      case 'failed': return 'status-error';
      default: return 'status-info';
    }
  };

  return (
    <div>
      {/* Header */}
      <div className="card">
        <div className="card-header">
          <h1 className="card-title">Model Training</h1>
          <p className="card-subtitle">
            {datasetId ? `Dataset: ${datasetId}` : 'Train machine learning models'}
          </p>
        </div>
      </div>

      {/* Training Configuration */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Training Configuration</h2>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="form-group">
            <label className="form-label">Target Column</label>
            <select
              className="form-select"
              value={selectedTargetColumn}
              onChange={(e) => setSelectedTargetColumn(e.target.value)}
            >
              <option value="">Select target column...</option>
              {datasetInfo?.columns?.map((col: string) => (
                <option key={col} value={col}>{col}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Task Type</label>
            <select
              className="form-select"
              value={taskType}
              onChange={(e) => setTaskType(e.target.value)}
            >
              <option value="auto">Auto-detect</option>
              <option value="classification">Classification</option>
              <option value="regression">Regression</option>
            </select>
          </div>
        </div>

        <button
          onClick={handleStartTraining}
          disabled={trainModelsMutation.isPending || trainingStatus?.status === 'training'}
          className="btn btn-primary"
        >
          {trainModelsMutation.isPending || trainingStatus?.status === 'training' ? (
            <>
              <div className="loading"></div>
              Training...
            </>
          ) : (
            'Start Training'
          )}
        </button>
      </div>

      {/* Training Progress */}
      {trainingStatus?.status === 'training' && (
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Training Progress</h2>
          </div>
          
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Current: {trainingStatus.current_model}</span>
                <span>{trainingStatus.progress}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${trainingStatus.progress}%` }}
                ></div>
              </div>
            </div>
            
            {trainingStatus.message && (
              <div className="text-sm text-gray-600">
                {trainingStatus.message}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Trained Models */}
      {models && models.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Trained Models</h2>
            <p className="card-subtitle">Model performance comparison</p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2 px-4">Model</th>
                  <th className="text-left py-2 px-4">Status</th>
                  <th className="text-left py-2 px-4">Accuracy/R²</th>
                  <th className="text-left py-2 px-4">F1/RMSE</th>
                  <th className="text-left py-2 px-4">Training Time</th>
                  <th className="text-left py-2 px-4">Actions</th>
                </tr>
              </thead>
              <tbody>
                {models.map((model) => (
                  <tr key={model.model_id} className="border-b hover:bg-gray-50">
                    <td className="py-3 px-4 font-medium">{model.model_name}</td>
                    <td className="py-3 px-4">
                      <span className={`status-badge ${getStatusColor(model.status)}`}>
                        {model.status}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      {model.status === 'completed' ? (
                        formatMetric(model.metrics?.test?.accuracy || model.metrics?.test?.r2)
                      ) : 'N/A'}
                    </td>
                    <td className="py-3 px-4">
                      {model.status === 'completed' ? (
                        formatMetric(model.metrics?.test?.f1_score || model.metrics?.test?.rmse)
                      ) : 'N/A'}
                    </td>
                    <td className="py-3 px-4">
                      {model.training_time ? `${model.training_time.toFixed(1)}s` : 'N/A'}
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex gap-2">
                        {model.status === 'completed' && (
                          <>
                            <button
                              onClick={() => navigate(`/predictions/${model.model_id}`)}
                              className="btn btn-primary btn-sm"
                            >
                              Predict
                            </button>
                            <button
                              onClick={() => {
                                // Navigate to explanations page
                                window.open(`${API_BASE_URL}/api/explanations/${model.model_id}/global`, '_blank');
                              }}
                              className="btn btn-secondary btn-sm"
                            >
                              Explain
                            </button>
                          </>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Model Training Info */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">About Model Training</h2>
        </div>
        <div className="space-y-4">
          <div>
            <h3 className="font-medium mb-2">🤖 Available Algorithms</h3>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• Random Forest - Robust ensemble method</li>
              <li>• LightGBM - Fast gradient boosting (if available)</li>
              <li>• Logistic Regression - Linear classification</li>
              <li>• Linear/Ridge Regression - Linear regression</li>
            </ul>
          </div>
          <div>
            <h3 className="font-medium mb-2">⚙️ Automatic Features</h3>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• Task type detection (classification/regression)</li>
              <li>• Hyperparameter optimization with Optuna</li>
              <li>• Cross-validation for robust evaluation</li>
              <li>• Feature preprocessing and encoding</li>
              <li>• Model comparison and selection</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Models;
