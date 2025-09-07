import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useMutation, useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface Dataset {
  dataset_id: string;
  filename: string;
  upload_time: string;
  file_size: number;
  status: string;
}

const Upload: React.FC = () => {
  const [uploadProgress, setUploadProgress] = useState(0);
  const navigate = useNavigate();

  // Fetch existing datasets
  const { data: datasets, refetch } = useQuery<Dataset[]>({
    queryKey: ['datasets'],
    queryFn: async () => {
      const response = await axios.get(`${API_BASE_URL}/api/upload/datasets`);
      return response.data;
    },
  });

  // Upload mutation
  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(`${API_BASE_URL}/api/upload/`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setUploadProgress(progress);
          }
        },
      });
      return response.data;
    },
    onSuccess: (data) => {
      toast.success('File uploaded successfully!');
      setUploadProgress(0);
      refetch();
      // Navigate to EDA page for the uploaded dataset
      navigate(`/eda/${data.dataset_id}`);
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Upload failed');
      setUploadProgress(0);
    },
  });

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      uploadMutation.mutate(file);
    }
  }, [uploadMutation]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
    },
    maxFiles: 1,
    maxSize: 100 * 1024 * 1024, // 100MB
  });

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  return (
    <div>
      <div className="card">
        <div className="card-header">
          <h1 className="card-title">Upload Dataset</h1>
          <p className="card-subtitle">
            Upload CSV, XLS, or XLSX files to start your analysis
          </p>
        </div>

        {/* Upload Area */}
        <div
          {...getRootProps()}
          className={`
            border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
            ${isDragActive ? 'border-blue-400 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}
            ${uploadMutation.isPending ? 'pointer-events-none opacity-50' : ''}
          `}
        >
          <input {...getInputProps()} />
          <div className="text-6xl mb-4">📊</div>
          {isDragActive ? (
            <p className="text-lg text-blue-600">Drop the file here...</p>
          ) : (
            <div>
              <p className="text-lg mb-2">Drag & drop a file here, or click to select</p>
              <p className="text-sm text-gray-500">
                Supports CSV, XLS, XLSX files up to 100MB
              </p>
            </div>
          )}
        </div>

        {/* Upload Progress */}
        {uploadMutation.isPending && (
          <div className="mt-4">
            <div className="flex justify-between text-sm mb-1">
              <span>Uploading...</span>
              <span>{uploadProgress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
          </div>
        )}
      </div>

      {/* Existing Datasets */}
      {datasets && datasets.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Your Datasets</h2>
            <p className="card-subtitle">Previously uploaded datasets</p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2 px-4">Filename</th>
                  <th className="text-left py-2 px-4">Size</th>
                  <th className="text-left py-2 px-4">Upload Time</th>
                  <th className="text-left py-2 px-4">Status</th>
                  <th className="text-left py-2 px-4">Actions</th>
                </tr>
              </thead>
              <tbody>
                {datasets.map((dataset) => (
                  <tr key={dataset.dataset_id} className="border-b hover:bg-gray-50">
                    <td className="py-3 px-4 font-medium">{dataset.filename}</td>
                    <td className="py-3 px-4">{formatFileSize(dataset.file_size)}</td>
                    <td className="py-3 px-4">{formatDate(dataset.upload_time)}</td>
                    <td className="py-3 px-4">
                      <span
                        className={`status-badge ${
                          dataset.status === 'completed'
                            ? 'status-success'
                            : dataset.status === 'processing'
                            ? 'status-warning'
                            : 'status-error'
                        }`}
                      >
                        {dataset.status}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex gap-2">
                        <button
                          onClick={() => navigate(`/eda/${dataset.dataset_id}`)}
                          className="btn btn-primary btn-sm"
                        >
                          View EDA
                        </button>
                        <button
                          onClick={() => navigate(`/models/${dataset.dataset_id}`)}
                          className="btn btn-secondary btn-sm"
                        >
                          Train Models
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Getting Started</h2>
        </div>
        <div className="space-y-4">
          <div>
            <h3 className="font-medium mb-2">📋 Data Requirements</h3>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• File formats: CSV, XLS, XLSX</li>
              <li>• Maximum file size: 100MB</li>
              <li>• First row should contain column headers</li>
              <li>• Include a target column for supervised learning</li>
            </ul>
          </div>
          <div>
            <h3 className="font-medium mb-2">🔄 What happens next?</h3>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• Automatic data validation and profiling</li>
              <li>• Initial exploratory data analysis</li>
              <li>• Data quality assessment</li>
              <li>• Ready for model training</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Upload;
