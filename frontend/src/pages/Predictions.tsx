import React, { useState } from 'react';
import { useParams } from 'react-router-dom';
import { useQuery, useMutation } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import axios from 'axios';

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
      const response = await axios.get(`${API_BASE_URL}/api/models/`);
      return response.data;
    },
  });

  // Fetch model schema
  const { data: modelSchema } = useQuery({
    queryKey: ['model-schema', selectedModel],
    queryFn: async () => {
      if (!selectedModel) throw new Error('Model ID is required');
      const response = await axios.get(`${API_BASE_URL}/api/predictions/${selectedModel}/schema`);
      return response.data;
    },
    enabled: !!selectedModel,
  });

  // Prediction mutation
  const predictMutation = useMutation({
    mutationFn: async (data: Record<string, any>) => {
      if (!selectedModel) throw new Error('Model ID is required');
      const response = await axios.post(`${API_BASE_URL}/api/predictions/${selectedModel}/predict`, {
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
    
    if (Object.keys(inputData).length === 0) {
      toast.error('Please provide input data');
      return;
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
            onChange={(e) => handleInputChange(fieldName, parseFloat(e.target.value) || 0)}
          />
          {field.description && (
            <p className="text-sm text-gray-500 mt-1">{field.description}</p>
          )}
        </div>
      );
    }
  };

  const selectedModelInfo = models?.find(m => m.model_id === selectedModel);

  return (
    <div>
      {/* Header */}
      <div className="card">
        <div className="card-header">
          <h1 className="card-title">Make Predictions</h1>
          <p className="card-subtitle">Use trained models to make predictions on new data</p>
        </div>
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
          <div className="mt-4 p-4 bg-blue-50 rounded-lg">
            <h3 className="font-medium mb-2">Model Information</h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium">Model:</span> {selectedModelInfo.model_name}
              </div>
              <div>
                <span className="font-medium">Task:</span> {selectedModelInfo.task_type}
              </div>
              <div>
                <span className="font-medium">Dataset:</span> {selectedModelInfo.dataset_id}
              </div>
              <div>
                <span className="font-medium">Target:</span> {selectedModelInfo.target_column}
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
            className="btn btn-primary mt-4"
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

          <div className="space-y-4">
            <div className="p-6 bg-green-50 rounded-lg text-center">
              <div className="text-3xl font-bold text-green-600 mb-2">
                {typeof predictionResult.prediction === 'number' 
                  ? predictionResult.prediction.toFixed(4)
                  : predictionResult.prediction}
              </div>
              <div className="text-sm text-gray-600">
                Predicted {selectedModelInfo?.target_column}
              </div>
            </div>

            {predictionResult.confidence && (
              <div className="p-4 bg-blue-50 rounded-lg">
                <div className="font-medium mb-2">Confidence Score</div>
                <div className="text-2xl font-bold text-blue-600">
                  {(predictionResult.confidence * 100).toFixed(1)}%
                </div>
              </div>
            )}

            {predictionResult.probability && (
              <div className="p-4 bg-purple-50 rounded-lg">
                <div className="font-medium mb-2">Class Probabilities</div>
                <div className="space-y-2">
                  {predictionResult.probability.map((prob, index) => (
                    <div key={index} className="flex justify-between">
                      <span>Class {index}</span>
                      <span className="font-medium">{(prob * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Batch Prediction */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Batch Predictions</h2>
          <p className="card-subtitle">Upload a CSV file for batch predictions</p>
        </div>

        <div className="text-center py-8 border-2 border-dashed border-gray-300 rounded-lg">
          <div className="text-4xl mb-4">📄</div>
          <p className="text-gray-600 mb-4">
            Upload a CSV file with the same features as the training data
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
        <div className="space-y-4">
          <div>
            <h3 className="font-medium mb-2">📝 Single Predictions</h3>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• Select a trained model from the dropdown</li>
              <li>• Fill in the required feature values</li>
              <li>• Click "Make Prediction" to get results</li>
              <li>• View confidence scores and explanations</li>
            </ul>
          </div>
          <div>
            <h3 className="font-medium mb-2">📊 Understanding Results</h3>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• Classification: Shows predicted class and probabilities</li>
              <li>• Regression: Shows predicted numerical value</li>
              <li>• Confidence scores indicate model certainty</li>
              <li>• Higher confidence = more reliable prediction</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Predictions;
