'use client';

import { useState, useRef, useEffect } from 'react';
import { ChatInterfaceProps } from '@/types';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  plot?: string;
}

interface AnalysisStep {
  id: string;
  title: string;
  description: string;
  action: string;
}

export default function ChatInterface({ file, fileId, onAnalysisRequest, analysisResults }: ChatInterfaceProps) {
  const [input, setInput] = useState('');
  const [ws, setWs] = useState<WebSocket | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  
  // Define analysis workflow steps
  const analysisSteps: AnalysisStep[] = [
    {
      id: 'summary',
      title: 'Dataset Overview',
      description: 'Basic statistics and information about your dataset',
      action: 'summary_statistics'
    },
    {
      id: 'distribution',
      title: 'Data Distribution',
      description: 'Visualize the distribution of values in your dataset',
      action: 'data_distribution'
    },
    {
      id: 'missing',
      title: 'Missing Values Analysis',
      description: 'Identify and analyze missing values in your dataset',
      action: 'missing_values'
    },
    {
      id: 'correlation',
      title: 'Correlation Analysis',
      description: 'Discover relationships between variables',
      action: 'correlation_analysis'
    },
    {
      id: 'outliers',
      title: 'Outlier Detection',
      description: 'Identify and analyze outliers in your dataset',
      action: 'outlier_detection'
    }
  ];

  const [quickActions] = useState([
    'Summary statistics',
    'Data distribution',
    'Missing values',
    'Correlation analysis',
    'Outlier detection'
  ]);

  useEffect(() => {
    // Connect to WebSocket
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws';
    const websocket = new WebSocket(wsUrl);
    
    websocket.onopen = () => {
      console.log('Connected to WebSocket');
      setIsConnected(true);
      setConnectionError(null);
      // No need to send file info here as we'll include file_id in each request
    };

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'error') {
        onAnalysisRequest(`Error: ${data.text}`);
      } else if (data.type === 'result') {
        // Create a response object with all the data from the server
        const responseData = {
          type: 'response',
          data: data,
          timestamp: new Date().toISOString()
        };
        onAnalysisRequest(JSON.stringify(responseData));
      } else if (data.type === 'info') {
        onAnalysisRequest(`Info: ${data.text}`);
      }
    };

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionError('Failed to connect to analysis server. Please try again later.');
      setIsConnected(false);
    };

    websocket.onclose = () => {
      setIsConnected(false);
    };

    setWs(websocket);

    return () => {
      websocket.close();
    };
  }, []);

  useEffect(() => {
    // Scroll to bottom when new results come in
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [analysisResults]);

  const sendQuickAction = (action: string) => {
    if (!ws || !fileId || !isConnected) return;
    const message = {
      action: 'run_analysis',
      function: action.toLowerCase().replace(/\s+/g, '_'),
      file_id: fileId
    };
    ws.send(JSON.stringify(message));
    onAnalysisRequest(`Requested ${action.toLowerCase()}`);
  };

  const handleCustomQuery = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !ws || !fileId || !isConnected) return;

    const message = {
      action: 'run_analysis',
      function: 'custom_query',
      query: input,
      file_id: fileId
    };

    ws.send(JSON.stringify(message));
    onAnalysisRequest(input);
    setInput('');
  };

  const goToNextStep = () => {
    if (currentStepIndex < analysisSteps.length - 1) {
      setCurrentStepIndex(currentStepIndex + 1);
      // Automatically trigger the analysis for this step
      sendQuickAction(analysisSteps[currentStepIndex + 1].action);
    }
  };

  const goToPreviousStep = () => {
    if (currentStepIndex > 0) {
      setCurrentStepIndex(currentStepIndex - 1);
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Connection Status */}
      {connectionError && (
        <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4">
          <p>{connectionError}</p>
        </div>
      )}

      {/* Analysis Steps Progress */}
      <div className="bg-white p-4 border-b shadow-sm">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold text-[#2A1458]">
            {analysisSteps[currentStepIndex].title}
          </h2>
          <div className="text-sm text-gray-500">
            Step {currentStepIndex + 1} of {analysisSteps.length}
          </div>
        </div>
        
        <p className="text-gray-600 mb-4">{analysisSteps[currentStepIndex].description}</p>
        
        <div className="flex space-x-2">
          <button
            onClick={goToPreviousStep}
            disabled={currentStepIndex === 0}
            className={`px-4 py-2 rounded-md ${currentStepIndex === 0 ? 'bg-gray-300 cursor-not-allowed' : 'bg-[#E8988A] hover:bg-[#D87A6C] text-white'}`}
          >
            Previous
          </button>
          <button
            onClick={() => sendQuickAction(analysisSteps[currentStepIndex].action)}
            disabled={!isConnected}
            className="px-4 py-2 bg-[#9B177E] text-white rounded-md hover:bg-[#7D1364] transition-colors flex-1"
          >
            Run Analysis
          </button>
          <button
            onClick={goToNextStep}
            disabled={currentStepIndex === analysisSteps.length - 1}
            className={`px-4 py-2 rounded-md ${currentStepIndex === analysisSteps.length - 1 ? 'bg-gray-300 cursor-not-allowed' : 'bg-[#2A1458] hover:bg-[#1A0D38] text-white'}`}
          >
            Next
          </button>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-[#FFEAD8] p-4 border-b">
        <h3 className="text-sm font-semibold text-gray-700 mb-3">Quick Analysis</h3>
        <div className="flex flex-wrap gap-2">
          {quickActions.map(action => (
            <button
              key={action}
              onClick={() => sendQuickAction(action)}
              disabled={!isConnected}
              className="px-3 py-1.5 text-sm bg-white rounded-full border border-[#E8988A] text-[#9B177E] hover:bg-[#E8988A] hover:text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {action}
            </button>
          ))}
        </div>
      </div>

      {/* Analysis Results */}
      <div className="flex-1 overflow-auto p-4 space-y-4 bg-gray-50">
        {analysisResults.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-500">
            <div className="text-center">
              <svg className="w-16 h-16 mx-auto text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <p className="mt-2">Run an analysis to see results here</p>
            </div>
          </div>
        ) : (
          analysisResults.map((result, index) => (
            <div
              key={index}
              className={`flex ${result.type === 'request' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] rounded-lg p-4 shadow-sm ${
                  result.type === 'request'
                    ? 'bg-[#9B177E] text-white'
                    : 'bg-white border border-gray-200'
                }`}
              >
                {typeof result.data === 'string' ? (
                  <p className="mb-2">{result.data}</p>
                ) : (
                  <div>
                    {result.data.text && <p className="mb-3">{result.data.text}</p>}
                    {result.data.stats && (
                      <div className="mb-3 p-3 bg-gray-50 rounded-md">
                        <h4 className="font-semibold mb-2">Statistics</h4>
                        <pre className="text-xs whitespace-pre-wrap overflow-x-auto">
                          {JSON.stringify(result.data.stats, null, 2)}
                        </pre>
                      </div>
                    )}
                    {result.data.plot && (
                      <div className="mt-3">
                        <img
                          src={`data:image/png;base64,${result.data.plot}`}
                          alt="Analysis visualization"
                          className="rounded-lg max-w-full border border-gray-200"
                        />
                      </div>
                    )}
                  </div>
                )}
                <div className="mt-2 text-xs text-gray-500 text-right">
                  {new Date(result.timestamp).toLocaleTimeString()}
                </div>
              </div>
            </div>
          ))
        )}
        <div ref={chatEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleCustomQuery} className="p-4 border-t bg-white shadow-md">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a custom question about your data..."
            className="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-[#9B177E]"
          />
          <button
            type="submit"
            className="px-6 py-2 bg-[#9B177E] text-white rounded-lg hover:bg-[#2A1458] transition-colors"
            disabled={!ws}
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
