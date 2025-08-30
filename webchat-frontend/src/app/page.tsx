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

  const handleFileUpload = (file: File) => {
    setSelectedDataset(file);
    setActiveFileId(file.name);
    // Additional logic for file upload will go here
  };

  const handleAnalysisRequest = (request: string) => {
    // Logic to send analysis request to backend will go here
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
        {selectedDataset && (
          <ChatInterface 
            file={selectedDataset}
            onAnalysisRequest={handleAnalysisRequest}
            analysisResults={analysisResults}
          />
        )}
      </div>
    </main>
  );
};

export default Home;
}
    </div>
  );
}
