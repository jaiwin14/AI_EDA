'use client';

import { useState } from 'react';
import { FileUploadProps } from '@/types';

export default function FileUpload({ onFileUploaded }: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

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
    const file = e.dataTransfer.files[0];
    if (file) {
      await uploadFile(file);
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      await uploadFile(file);
    }
  };

  const uploadFile = async (file: File) => {
    if (!file.name.match(/\.(csv|xlsx)$/i)) {
      alert('Please upload a CSV or Excel file');
      return;
    }

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Upload failed');

      const data = await response.json();
      // Pass both file and file_id to parent component
      onFileUploaded(file, data.file_id);
    } catch (error) {
      console.error('Upload error:', error);
      alert('Failed to upload file. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div
      className={`w-full max-w-xl p-8 border-2 border-dashed rounded-lg ${
        isDragging
          ? 'border-[#9B177E] bg-[#FFEAD8]'
          : 'border-gray-300 hover:border-[#E8988A]'
      } transition-colors cursor-pointer text-center`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => document.getElementById('fileInput')?.click()}
    >
      <input
        type="file"
        id="fileInput"
        accept=".csv,.xlsx"
        className="hidden"
        onChange={handleFileSelect}
      />
      {isUploading ? (
        <div className="text-[#9B177E]">Uploading...</div>
      ) : (
        <>
          <div className="text-4xl mb-4">📊</div>
          <p className="text-gray-600 mb-2">
            Drag and drop your data file here, or click to browse
          </p>
          <p className="text-sm text-gray-500">Supported formats: .csv, .xlsx</p>
        </>
      )}
    </div>
  );
}
