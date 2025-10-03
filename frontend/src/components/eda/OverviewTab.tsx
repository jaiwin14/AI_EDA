import { useQuery } from '@tanstack/react-query';
import aiService from '../../services/aiService';

interface OverviewTabProps {
  datasetId: string;
  basicStats: any;
  missingValues: any;
}

const OverviewTab: React.FC<OverviewTabProps> = ({ datasetId, basicStats, missingValues }) => {
  // Fetch AI overview using new AI service
  const { data: aiOverview, isLoading: aiOverviewLoading, error: aiOverviewError } = useQuery({
    queryKey: ['ai-overview', datasetId],
    queryFn: async () => {
      if (!datasetId) throw new Error('Dataset ID is required');
      console.log('Fetching AI overview for dataset:', datasetId);
      const result = await aiService.getDatasetOverview(datasetId);
      console.log('AI Overview result:', result);
      return result;
    },
    enabled: !!datasetId,
    retry: 1,
  });

  return (
    <div>
      {/* AI Overview Section */}
      {aiOverviewLoading && (
        <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-center">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
            <span className="text-blue-800">Loading AI insights...</span>
          </div>
        </div>
      )}
      
      {aiOverviewError && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-800">Failed to load AI overview: {String(aiOverviewError)}</p>
        </div>
      )}
      
      {aiOverview?.data?.overview && (
        <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <h3 className="font-semibold mb-3 text-blue-800 flex items-center">
            <span className="mr-2">🤖</span>
            AI Dataset Overview
          </h3>
          <div className="space-y-2">
            <p className="text-blue-900 font-medium">
              {aiOverview.data.overview.overview_line_1}
            </p>
            <p className="text-blue-800">
              {aiOverview.data.overview.overview_line_2}
            </p>
            {aiOverview.data.overview.key_characteristics && (
              <div className="mt-3">
                <h4 className="font-medium text-blue-800 mb-2">Key Characteristics:</h4>
                <ul className="list-disc list-inside space-y-1">
                  {aiOverview.data.overview.key_characteristics.map((characteristic: string, index: number) => (
                    <li key={index} className="text-blue-700 text-sm">{characteristic}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}
      
      {/* Parse AI Overview Data */}
      {(() => {
        let overviewData = null;
        
        if (aiOverview?.data?.overview) {
          overviewData = aiOverview.data.overview;
        } else if (aiOverview?.data?.insights) {
          try {
            const insightsStr = aiOverview.data.insights;
            const jsonMatch = insightsStr.match(/```json\n([\s\S]*?)\n```/);
            if (jsonMatch) {
              const parsedData = JSON.parse(jsonMatch[1]);
              overviewData = parsedData?.overview;
            }
          } catch (e) {
            console.error('Failed to parse AI overview insights:', e);
          }
        }
        
        return overviewData && (
          <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <h3 className="font-semibold mb-3 text-blue-800 flex items-center">
              <span className="mr-2">🤖</span>
              AI Dataset Overview
            </h3>
            <div className="space-y-2">
              <p className="text-blue-900 font-medium">
                {overviewData.overview_line_1}
              </p>
              <p className="text-blue-800">
                {overviewData.overview_line_2}
              </p>
              {overviewData.key_characteristics && (
                <div className="mt-3">
                  <h4 className="font-medium text-blue-800 mb-2">Key Characteristics:</h4>
                  <ul className="list-disc list-inside space-y-1">
                    {overviewData.key_characteristics.map((characteristic: string, index: number) => (
                      <li key={index} className="text-blue-700 text-sm">{characteristic}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        );
      })()}
      
      {/* Fallback: Show raw AI data if structure is different */}
      {aiOverview && !(() => {
        if (aiOverview?.data?.overview) return true;
        if (aiOverview?.data?.insights) {
          try {
            const insightsStr = aiOverview.data.insights;
            const jsonMatch = insightsStr.match(/```json\n([\s\S]*?)\n```/);
            if (jsonMatch) {
              const parsedData = JSON.parse(jsonMatch[1]);
              return parsedData?.overview;
            }
          } catch (e) {
            // ignore
          }
        }
        return false;
      })() && (
        <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <h3 className="font-semibold mb-3 text-yellow-800 flex items-center">
            <span className="mr-2">🤖</span>
            AI Dataset Overview (Raw)
          </h3>
          <pre className="text-xs text-yellow-900 bg-yellow-100 p-2 rounded overflow-auto">
            {JSON.stringify(aiOverview, null, 2)}
          </pre>
        </div>
      )}
      
      <h3 className="font-semibold mb-4">Data Quality Summary</h3>
      {missingValues.columns_with_missing && 
       Object.keys(missingValues.columns_with_missing).length > 0 ? (
        <div className="mb-4">
          <h4 className="font-medium mb-2">Columns with Missing Values</h4>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2">Column</th>
                  <th className="text-left py-2">Missing Count</th>
                  <th className="text-left py-2">Missing %</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(missingValues.columns_with_missing).map(([col, count]) => (
                  <tr key={col} className="border-b">
                    <td className="py-2">{col}</td>
                    <td className="py-2">{count as number}</td>
                    <td className="py-2">
                      {((count as number / basicStats.total_rows) * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="text-green-600 mb-4">✅ No missing values detected!</div>
      )}
    </div>
  );
};

export default OverviewTab;
