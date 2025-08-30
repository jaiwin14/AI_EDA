'use client';

import { useState, useRef, useEffect } from 'react';
import { ChatInterfaceProps } from '@/types';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  plot?: string;
}

export default function ChatInterface({ file, fileId, onAnalysisRequest, analysisResults }: ChatInterfaceProps) {
  const [input, setInput] = useState('');
  const [ws, setWs] = useState<WebSocket | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const [quickActions] = useState([
    'Summary statistics',
    'Data distribution',
    'Missing values',
    'Correlation analysis',
    'Outlier detection'
  ]);

  useEffect(() => {
    // Connect to WebSocket
    const websocket = new WebSocket('ws://localhost:8000/ws');
    
    websocket.onopen = () => {
      console.log('Connected to WebSocket');
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
    if (!ws || !fileId) return;
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
    if (!input.trim() || !ws || !fileId) return;

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

  return (
    <div className="h-full flex flex-col">
      {/* Quick Actions */}
      <div className="bg-[#FFEAD8] p-4 border-b">
        <h3 className="text-sm font-semibold text-gray-700 mb-3">Quick Analysis</h3>
        <div className="flex flex-wrap gap-2">
          {quickActions.map(action => (
            <button
              key={action}
              onClick={() => sendQuickAction(action)}
              className="px-3 py-1.5 text-sm bg-white rounded-full border border-[#E8988A] text-[#9B177E] hover:bg-[#E8988A] hover:text-white transition-colors"
            >
              {action}
            </button>
          ))}
        </div>
      </div>

      {/* Analysis Results */}
      <div className="flex-1 overflow-auto p-4 space-y-4">
        {analysisResults.map((result, index) => (
          <div
            key={index}
            className={`flex ${result.type === 'request' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] rounded-lg p-4 ${
                result.type === 'request'
                  ? 'bg-[#9B177E] text-white'
                  : 'bg-[#FFEAD8]'
              }`}
            >
              {typeof result.data === 'string' ? (
                <p className="mb-2">{result.data}</p>
              ) : (
                <pre className="whitespace-pre-wrap overflow-x-auto">
                  {JSON.stringify(result.data, null, 2)}
                </pre>
              )}
              {result.type === 'response' && result.data?.plot && (
                <img
                  src={`data:image/png;base64,${result.data.plot}`}
                  alt="Analysis visualization"
                  className="mt-2 rounded-lg max-w-full"
                />
              )}
            </div>
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleCustomQuery} className="p-4 border-t bg-white">
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
