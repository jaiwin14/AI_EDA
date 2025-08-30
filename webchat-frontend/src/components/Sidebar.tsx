'use client';

import { SidebarProps } from '@/types';

export default function Sidebar({ activeFileId }: SidebarProps) {
  return (
    <div className="w-64 bg-[#2A1458] text-white p-4 flex flex-col min-h-screen">
      <h1 className="text-xl font-bold mb-6">AI EDA</h1>
      
      {/* Active File */}
      {activeFileId && (
        <div className="mb-4">
          <h2 className="text-sm font-semibold text-[#E8988A] uppercase mb-2">Active File</h2>
          <div className="p-2 bg-[#9B177E] rounded-md">
            <p className="text-sm truncate">{activeFileId}</p>
          </div>
        </div>
      )}

      {/* Analysis History */}
      <div className="mt-auto">
        <h2 className="text-sm font-semibold text-[#E8988A] uppercase mb-2">Tips</h2>
        <ul className="text-sm space-y-2 text-gray-300">
          <li>• Upload CSV or Excel files</li>
          <li>• Use quick actions for common analysis</li>
          <li>• Ask custom questions about your data</li>
          <li>• View visualizations and insights</li>
        </ul>
      </div>

      {/* Credits */}
      <div className="mt-4 pt-4 border-t border-gray-700">
        <p className="text-xs text-gray-400">
          Made with 💜 using Next.js & FastAPI
        </p>
      </div>
    </div>
  );
}
