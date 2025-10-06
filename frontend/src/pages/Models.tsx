import React, { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import axios from 'axios';
import './Models.css'; // Make sure to import the CSS file

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
      const response = await axios.get(`${API_BASE_URL}/api/v1/upload/datasets/${datasetId}`);
      return response.data;
    },
    enabled: !!datasetId,
  });

  // Fetch trained models
  const { data: models, refetch: refetchModels } = useQuery<ModelResult[]>({
    queryKey: ['models', datasetId],
    queryFn: async () => {
      if (!datasetId) throw new Error('Dataset ID is required');
      const response = await axios.get(`${API_BASE_URL}/api/v1/models/${datasetId}`);
      return response.data;
    },
    enabled: !!datasetId,
  });

  // Fetch training status
  const { data: trainingStatus } = useQuery<TrainingStatus>({
    queryKey: ['training-status', datasetId],
    queryFn: async () => {
      if (!datasetId) throw new Error('Dataset ID is required');
      const response = await axios.get(`${API_BASE_URL}/api/v1/models/${datasetId}/status`);
      return response.data;
    },
    enabled: !!datasetId,
    refetchInterval: (data) => data?.status === 'training' ? 2000 : false,
  });

  // Start training mutation
  const trainModelsMutation = useMutation({
    mutationFn: async (params: { target_column: string; task_type: string }) => {
      if (!datasetId) throw new Error('Dataset ID is required');
      const response = await axios.post(`${API_BASE_URL}/api/v1/models/train`, { dataset_id: datasetId, ...params });
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
    <div className="main-content">
      {/* Header */}
      <div className="page-header">
        <h1 className="page-title">Model Training</h1>
        <p className="page-subtitle">
          {datasetId ? `Dataset: ${datasetId}` : 'Train machine learning models'}
        </p>
      </div>

      {/* Training Configuration */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Training Configuration</h2>
        </div>
        
        <div className="grid grid-cols-2">
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
          
          <div className="progress-container">
            <div>
              <div className="progress-header">
                <span>Current Model: {trainingStatus.current_model}</span>
                <span>{trainingStatus.progress}%</span>
              </div>
              <div className="progress-bar-bg">
                <div
                  className="progress-bar-fg"
                  style={{ width: `${trainingStatus.progress}%` }}
                ></div>
              </div>
            </div>
            
            {trainingStatus.message && (
              <div className="progress-message">
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

          <div className="table-container">
            <table className="results-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Status</th>
                  <th>Accuracy/R²</th>
                  <th>F1/RMSE</th>
                  <th>Training Time</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {models.map((model) => (
                  <tr key={model.model_id}>
                    <td>{model.model_name}</td>
                    <td>
                      <span className={`status-badge ${getStatusColor(model.status)}`}>
                        {model.status}
                      </span>
                    </td>
                    <td>
                      {model.status === 'completed' ? (
                        formatMetric(model.metrics?.test?.accuracy || model.metrics?.test?.r2)
                      ) : 'N/A'}
                    </td>
                    <td>
                      {model.status === 'completed' ? (
                        formatMetric(model.metrics?.test?.f1_score || model.metrics?.test?.rmse)
                      ) : 'N/A'}
                    </td>
                    <td>
                      {model.training_time ? `${model.training_time.toFixed(1)}s` : 'N/A'}
                    </td>
                    <td>
                      <div className="actions-group">
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
                                window.open(`${API_BASE_URL}/api/v1/explanations/${model.model_id}/global`, '_blank');
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
        <div className="grid grid-cols-2">
          <div className="about-box">
            <h3 className="about-title">
              <span className="about-icon">🤖</span>
              Available Algorithms
            </h3>
            <ul className="about-list">
              <li>Random Forest - Robust ensemble method</li>
              <li>LightGBM - Fast gradient boosting</li>
              <li>Logistic Regression - Linear classification</li>
              <li>Linear/Ridge Regression - Linear regression</li>
            </ul>
          </div>
          <div className="about-box">
            <h3 className="about-title">
              <span className="about-icon">⚙️</span>
              Automatic Features
            </h3>
            <ul className="about-list">
              <li>Task type detection (classification/regression)</li>
              <li>Hyperparameter optimization with Optuna</li>
              <li>Cross-validation for robust evaluation</li>
              <li>Feature preprocessing and encoding</li>
              <li>Model comparison and selection</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Models;