import React, { useState } from 'react';
import { useParams } from 'react-router-dom';
import { useQuery, useMutation } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import axios from 'axios';
import './Predictions.css'; // Import the new CSS file

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

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

const Predictions: React.FC = () => {
  const { modelId } = useParams<{ modelId: string }>();
  const [selectedModel, setSelectedModel] = useState<string>(modelId || '');
  const [inputData, setInputData] = useState<Record<string, any>>({});
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);

  // Fetch available models
  const { data: models } = useQuery<Model[]>({
    queryKey: ['all-models'],
    queryFn: async () => {
      const response = await axios.get(`${API_BASE_URL}/api/v1/models/`);
      return response.data;
    },
  });

  // Fetch model schema
  const { data: modelSchema } = useQuery({
    queryKey: ['model-schema', selectedModel],
    queryFn: async () => {
      if (!selectedModel) throw new Error('Model ID is required');
      const response = await axios.get(`${API_BASE_URL}/api/v1/predictions/${selectedModel}/schema`);
      return response.data;
    },
    enabled: !!selectedModel,
    onSuccess: () => {
        // Reset form and results when model changes
        setInputData({});
        setPredictionResult(null);
    }
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
    
    // Check if all fields in the schema are filled
    const requiredFields = modelSchema?.features?.map((f: any) => f.name) || [];
    for (const field of requiredFields) {
        if (inputData[field] === undefined || inputData[field] === '') {
            toast.error(`Please provide a value for ${field}`);
            return;
        }
    }

    predictMutation.mutate(inputData);
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

  const selectedModelInfo = models?.find(m => m.model_id === selectedModel);

  return (
    <div className="main-content">
      {/* Header */}
      <div className="page-header">
        <h1 className="page-title">Make Predictions</h1>
        <p className="page-subtitle">Use trained models to make predictions on new data</p>
      </div>

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
            {models?.filter(m => m.status === 'completed').map((model) => (
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

      {/* Batch Prediction */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Batch Predictions</h2>
          <p className="card-subtitle">Upload a CSV file for batch predictions</p>
        </div>

        <div className="batch-upload-box">
          <div className="batch-upload-icon">📄</div>
          <p>
            Upload a CSV file with the same features as the training data.
          </p>
          <button className="btn btn-secondary" disabled>
            Upload CSV (Coming Soon)
          </button>
        </div>
      </div>

      {/* Instructions */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">How to Use</h2>
        </div>
        <div className="grid grid-cols-2">
          <div className="about-box">
            <h3 className="about-title">
              <span className="about-icon">📝</span>
              Single Predictions
            </h3>
            <ul className="about-list">
              <li>Select a trained model from the dropdown.</li>
              <li>Fill in the required feature values.</li>
              <li>Click "Make Prediction" to get results.</li>
              <li>View confidence scores and explanations.</li>
            </ul>
          </div>
          <div className="about-box">
            <h3 className="about-title">
              <span className="about-icon">📊</span>
              Understanding Results
            </h3>
            <ul className="about-list">
              <li>Classification: Shows predicted class and probabilities.</li>
              <li>Regression: Shows predicted numerical value.</li>
              <li>Confidence scores indicate model certainty.</li>
              <li>Higher confidence = more reliable prediction.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Predictions;
