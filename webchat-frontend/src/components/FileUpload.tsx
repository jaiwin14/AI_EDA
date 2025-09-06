'use client';

import { useState } from 'react';
import { FileUploadProps } from '@/types';

export default function FileUpload({ onFileUploaded }: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    setError(null);
    const file = e.dataTransfer.files[0];
    if (file) {
      await uploadFile(file);
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setError(null);
      await uploadFile(file);
    }
  };

  const uploadFile = async (file: File) => {
    // Validate file type
    if (!file.name.match(/\.(csv|xlsx|xls)$/i)) {
      setError('Please upload a CSV or Excel file (.csv, .xlsx, .xls)');
      return;
    }

    // Validate file size (100MB limit)
    if (file.size > 100 * 1024 * 1024) {
      setError('File size must be less than 100MB');
      return;
    }

    setIsUploading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Upload failed with status ${response.status}`);
      }

      const data = await response.json();
      onFileUploaded(file, data.file_id);
    } catch (error) {
      console.error('Upload error:', error);
      setError(error instanceof Error ? error.message : 'Failed to upload file. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div
        className={`w-full p-8 border-2 border-dashed rounded-xl transition-all duration-300 cursor-pointer text-center ${
          isDragging
            ? 'border-blue-500 bg-blue-50 shadow-lg'
            : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
        } ${error ? 'border-red-300 bg-red-50' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => document.getElementById('fileInput')?.click()}
      >
        <input
          type="file"
          id="fileInput"
          accept=".csv,.xlsx,.xls"
          className="hidden"
          onChange={handleFileSelect}
        />
        
        {isUploading ? (
          <div className="space-y-4">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
            <div className="text-blue-600 font-medium">Uploading your file...</div>
            <div className="text-sm text-gray-500">Please wait while we process your data</div>
          </div>
        ) : error ? (
          <div className="space-y-4">
            <div className="text-4xl">⚠️</div>
            <div className="text-red-600 font-medium">Upload Error</div>
            <div className="text-sm text-red-500">{error}</div>
            <div className="text-xs text-gray-500">Click to try again</div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="text-6xl mb-4">📊</div>
            <div className="space-y-2">
              <h3 className="text-xl font-semibold text-gray-800">
                Upload Your Data File
              </h3>
              <p className="text-gray-600">
                Drag and drop your data file here, or click to browse
              </p>
              <p className="text-sm text-gray-500">
                Supported formats: CSV, Excel (.xlsx, .xls)
              </p>
              <p className="text-xs text-gray-400">
                Maximum file size: 100MB
              </p>
            </div>
          </div>
        )}
      </div>
      
      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center space-x-2">
            <div className="text-red-500">⚠️</div>
            <div className="text-sm text-red-700">{error}</div>
          </div>
        </div>
      )}
    </div>
  );
}
