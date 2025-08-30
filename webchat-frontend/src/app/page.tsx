'use client';

import { useState } from 'react';
import dynamic from 'next/dynamic';
import type { AnalysisResult } from '@/types';

// Dynamic imports to avoid SSR issues with WebSocket
const FileUpload = dynamic(() => import('@/components/FileUpload'), { ssr: false });
const ChatInterface = dynamic(() => import('@/components/ChatInterface'), { ssr: false });
const Sidebar = dynamic(() => import('@/components/Sidebar'), { ssr: false });

const Home = () => {
  const [selectedDataset, setSelectedDataset] = useState<File | null>(null);
  const [activeFileId, setActiveFileId] = useState<string | null>(null);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult[]>([]);

  const handleFileUpload = (file: File, fileId: string) => {
    setSelectedDataset(file);
    setActiveFileId(fileId);
    // Clear previous analysis results when a new file is uploaded
    setAnalysisResults([]);
  };

  const handleAnalysisRequest = (request: string) => {
    try {
      // Check if the request is a JSON string (response from server)
      const parsedData = JSON.parse(request);
      if (parsedData.type === 'response') {
        // It's a response from the server
        setAnalysisResults(prev => [...prev, parsedData]);
        return;
      }
    } catch (e) {
      // Not a JSON string, treat as a regular request
    }
    
    // Handle as a regular request
    const newResult: AnalysisResult = {
      type: 'request',
      data: request,
      timestamp: new Date().toISOString()
    };
    setAnalysisResults(prev => [...prev, newResult]);
  };

  return (
    <main className="flex h-screen bg-[#FFEAD8]">
      <Sidebar activeFileId={activeFileId} />
      <div className="flex-1 flex flex-col">
        <FileUpload onFileUploaded={handleFileUpload} />
        {selectedDataset && activeFileId && (
          <ChatInterface 
            file={selectedDataset}
            fileId={activeFileId}
            onAnalysisRequest={handleAnalysisRequest}
            analysisResults={analysisResults}
          />
        )}
      </div>
    </main>
  );
};

export default Home;
