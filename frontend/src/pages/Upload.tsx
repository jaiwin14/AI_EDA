import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useMutation, useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import axios from 'axios';
import './Upload.css';

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

  const { data: datasets, refetch } = useQuery<Dataset[]>({
    queryKey: ['datasets'],
    queryFn: async () => {
      const response = await axios.get(`${API_BASE_URL}/api/v1/upload/datasets`);
      return response.data;
    },
  });

  const uploadMutation = useMutation<any, Error, File>({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(`${API_BASE_URL}/api/v1/upload/`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
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
    maxSize: 100 * 1024 * 1024,
  });

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' });
  };

  return (
    <div className="upload-page-layout">
      {/* --- Left Column --- */}
      <div className="left-column">
        <header className="upload-header">
          <h1>Upload Dataset</h1>
          <p>Drag and drop your file below to begin the analysis.</p>
        </header>

        {/* The variables from useDropzone are used here */}
        <div
          {...getRootProps()}
          className={`upload-area ${isDragActive ? 'is-active' : ''} ${
            uploadMutation.isPending ? 'is-uploading' : ''
          }`}
        >
          <input {...getInputProps()} />
          <div className="upload-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
          </div>
          {isDragActive ? (
            <p className="upload-text-main is-active">Drop the file here...</p>
          ) : (
            <div>
              <p className="upload-text-main">Click to browse or drag a file here</p>
              <p className="upload-text-hint">Supports CSV, XLS, XLSX up to 100MB</p>
            </div>
          )}
        </div>

        {uploadMutation.isPending && (
          <div className="progress-container">
            <div className="progress-labels">
              <span>Uploading...</span>
              <span>{uploadProgress}%</span>
            </div>
            <div className="progress-bar">
              <div
                className="progress-bar-inner"
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
          </div>
        )}

        <footer className="minimal-instructions">
          <h3>Quick Guide</h3>
          <p>
            For best results, ensure your file's first row contains column headers.
            Including a target column is required for supervised learning models.
            After upload, you'll be redirected to the automated EDA report.
          </p>
        </footer>
      </div>

      {/* --- Right Column --- */}
      <div className="right-column">
        {datasets && datasets.length > 0 && (
          <section className="datasets-section">
            <header className="datasets-header">
              <h2>Your Datasets</h2>
            </header>
            <div className="table-container">
              <table className="datasets-table">
                <thead>
                  <tr>
                    <th>Filename</th>
                    <th>Size</th>
                    <th>Upload Time</th>
                    <th>Status</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {datasets.map((dataset) => (
                    <tr key={dataset.dataset_id}>
                      <td className="font-medium">{dataset.filename}</td>
                      <td>{formatFileSize(dataset.file_size)}</td>
                      <td>{formatDate(dataset.upload_time)}</td>
                      <td>
                        <span
                          className={`status-badge ${
                            dataset.status === 'completed' ? 'status-success'
                            : dataset.status === 'processing' ? 'status-warning'
                            : 'status-error'
                          }`}
                        >
                          {dataset.status}
                        </span>
                      </td>
                      <td>
                        <div className="table-actions">
                          <button
                            onClick={() => navigate(`/eda/${dataset.dataset_id}`)}
                            className="btn btn-primary btn-sm"
                          >
                            View EDA
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}
      </div>
    </div>
  );
};

export default Upload;