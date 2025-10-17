import React, { useState } from 'react';
import { useParams } from 'react-router-dom';
import { useQuery, useMutation } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import axios from 'axios';
import './MLWorkflow.css';

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

interface Model {
  model_id: string;
  model_name: string;
  dataset_id: string;
  target_column: string;
  task_type: string;
  status: string;
}

interface PredictionResult {
  prediction: number | string;
  probability?: number[];
  confidence?: number;
  explanation?: any;
}

interface TrainingStatus {
  status: string;
  progress: number;
  current_model: string;
  message: string;
}

const MLWorkflow: React.FC = () => {
  const { datasetId } = useParams<{ datasetId: string }>();

  // Tab state
  const [activeTab, setActiveTab] = useState<'training' | 'predictions'>('training');

  // Training state
  const [selectedTargetColumn, setSelectedTargetColumn] = useState<string>('');
  const [taskType, setTaskType] = useState<string>('auto');

  // Prediction state
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [inputData, setInputData] = useState<Record<string, any>>({});
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);

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

  // Fetch all available models for predictions
  const { data: allModels } = useQuery<Model[]>({
    queryKey: ['all-models'],
    queryFn: async () => {
      const response = await axios.get(`${API_BASE_URL}/api/v1/models/`);
      return response.data;
    },
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

  // Fetch model schema for predictions
  const { data: modelSchema } = useQuery({
    queryKey: ['model-schema', selectedModel],
    queryFn: async () => {
      if (!selectedModel) throw new Error('Model ID is required');
      const response = await axios.get(`${API_BASE_URL}/api/v1/predictions/${selectedModel}/schema`);
      return response.data;
    },
    enabled: !!selectedModel && activeTab === 'predictions',
    onSuccess: () => {
      setInputData({});
      setPredictionResult(null);
    }
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

  // Prediction mutation
  const predictMutation = useMutation({
    mutationFn: async (data: Record<string, any>) => {
      if (!selectedModel) throw new Error('Model ID is required');
      const response = await axios.post(`${API_BASE_URL}/api/v1/predictions/${selectedModel}/predict`, {
        input_data: data,
      });
      return response.data;
    },
    onSuccess: (data) => {
      setPredictionResult(data);
      toast.success('Prediction completed!');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Prediction failed');
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

  const handleInputChange = (field: string, value: any) => {
    setInputData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handlePredict = () => {
    if (!selectedModel) {
      toast.error('Please select a model');
      return;
    }

    const requiredFields = modelSchema?.features?.map((f: any) => f.name) || [];
    for (const field of requiredFields) {
      if (inputData[field] === undefined || inputData[field] === '') {
        toast.error(`Please provide a value for ${field}`);
        return;
      }
    }

    predictMutation.mutate(inputData);
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

  const renderInputField = (field: any) => {
    const fieldName = field.name;
    const fieldType = field.type;

    if (fieldType === 'categorical' && field.categories) {
      return (
        <div key={fieldName} className="form-group">
          <label className="form-label">{fieldName}</label>
          <select
            className="form-select"
            value={inputData[fieldName] || ''}
            onChange={(e) => handleInputChange(fieldName, e.target.value)}
          >
            <option value="">Select {fieldName}...</option>
            {field.categories.map((category: string) => (
              <option key={category} value={category}>{category}</option>
            ))}
          </select>
        </div>
      );
    } else {
      return (
        <div key={fieldName} className="form-group">
          <label className="form-label">{fieldName}</label>
          <input
            type="number"
            className="form-input"
            placeholder={`Enter ${fieldName}`}
            value={inputData[fieldName] || ''}
            onChange={(e) => handleInputChange(fieldName, parseFloat(e.target.value) || '')}
          />
          {field.description && (
            <p className="form-field-description">{field.description}</p>
          )}
        </div>
      );
    }
  };

  const selectedModelInfo = allModels?.find(m => m.model_id === selectedModel);

  return (
    <div className="ml-workflow-container">
      {/* Header */}
      <div className="page-header">
        <h1 className="page-title">Machine Learning Workflow</h1>
        <p className="page-subtitle">
          {datasetId ? `Dataset: ${datasetId}` : 'Train models and make predictions'}
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="tab-navigation">
        <button
          className={`tab-button ${activeTab === 'training' ? 'active' : ''}`}
          onClick={() => setActiveTab('training')}
        >
          🤖 Model Training
        </button>
        <button
          className={`tab-button ${activeTab === 'predictions' ? 'active' : ''}`}
          onClick={() => setActiveTab('predictions')}
        >
          🔮 Predictions
        </button>
      </div>

      {/* Training Tab */}
      {activeTab === 'training' && (
        <div className="tab-content">
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
                              <button
                                onClick={() => {
                                  setActiveTab('predictions');
                                  setSelectedModel(model.model_id);
                                }}
                                className="btn btn-primary btn-sm"
                              >
                                Predict
                              </button>
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
      )}

      {/* Predictions Tab */}
      {activeTab === 'predictions' && (
        <div className="tab-content">
          {/* Model Selection */}
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Select Model</h2>
            </div>

            <div className="form-group">
              <label className="form-label">Available Models</label>
              <select
                className="form-select"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
              >
                <option value="">Select a model...</option>
                {allModels?.filter(m => m.status === 'completed').map((model) => (
                  <option key={model.model_id} value={model.model_id}>
                    {model.model_name} - {model.dataset_id} ({model.task_type})
                  </option>
                ))}
              </select>
            </div>

            {selectedModelInfo && (
              <div className="model-info-box">
                <h3 className="model-info-title">Model Information</h3>
                <div className="grid grid-cols-2">
                  <div>
                    <span className="info-label">Model:</span> {selectedModelInfo.model_name}
                  </div>
                  <div>
                    <span className="info-label">Task:</span> {selectedModelInfo.task_type}
                  </div>
                  <div>
                    <span className="info-label">Dataset:</span> {selectedModelInfo.dataset_id}
                  </div>
                  <div>
                    <span className="info-label">Target:</span> {selectedModelInfo.target_column}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Input Form */}
          {modelSchema && (
            <div className="card">
              <div className="card-header">
                <h2 className="card-title">Input Data</h2>
                <p className="card-subtitle">Provide values for the following features</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {modelSchema.features?.map((field: any) => renderInputField(field))}
              </div>

              <button
                onClick={handlePredict}
                disabled={predictMutation.isPending}
                className="btn btn-primary"
                style={{ marginTop: '1.5rem' }}
              >
                {predictMutation.isPending ? (
                  <>
                    <div className="loading"></div>
                    Predicting...
                  </>
                ) : (
                  'Make Prediction'
                )}
              </button>
            </div>
          )}

          {/* Prediction Results */}
          {predictionResult && (
            <div className="card">
              <div className="card-header">
                <h2 className="card-title">Prediction Result</h2>
              </div>

              <div className="results-grid">
                <div className="prediction-main-result">
                  <div className="prediction-value">
                    {typeof predictionResult.prediction === 'number'
                      ? predictionResult.prediction.toFixed(4)
                      : predictionResult.prediction}
                  </div>
                  <div className="prediction-label">
                    Predicted {selectedModelInfo?.target_column}
                  </div>
                </div>

                <div className="results-details">
                  {predictionResult.confidence && (
                    <div className="prediction-extra-info">
                      <div className="info-label">Confidence Score</div>
                      <div className="confidence-value">
                        {(predictionResult.confidence * 100).toFixed(1)}%
                      </div>
                    </div>
                  )}

                  {predictionResult.probability && (
                    <div className="prediction-extra-info">
                      <div className="info-label">Class Probabilities</div>
                      <div className="probabilities-list">
                        {predictionResult.probability.map((prob, index) => (
                          <div key={index} className="probability-item">
                            <span>Class {index}</span>
                            <span>{(prob * 100).toFixed(1)}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Instructions */}
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">How to Use</h2>
            </div>
            <div className="grid grid-cols-2">
              <div className="about-box">
                <h3 className="about-title">
                  <span className="about-icon">🎯</span>
                  Complete Workflow
                </h3>
                <ul className="about-list">
                  <li>Train models in the Training tab</li>
                  <li>Switch to Predictions tab to use trained models</li>
                  <li>Fill in feature values and get predictions</li>
                  <li>View confidence scores and results</li>
                </ul>
              </div>
              <div className="about-box">
                <h3 className="about-title">
                  <span className="about-icon">📊</span>
                  Understanding Results
                </h3>
                <ul className="about-list">
                  <li>Classification: Shows predicted class and probabilities</li>
                  <li>Regression: Shows predicted numerical value</li>
                  <li>Confidence scores indicate model certainty</li>
                  <li>Higher confidence = more reliable prediction</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MLWorkflow;
