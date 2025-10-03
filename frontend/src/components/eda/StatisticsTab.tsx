import { useQuery } from '@tanstack/react-query';
import aiService from '../../services/aiService';
import NumericalVariableCard from './NumericalVariableCard';
import CategoricalVariableCard from './CategoricalVariableCard';

interface StatisticsTabProps {
  datasetId: string;
  statisticalAnalysis: any;
}

const StatisticsTab: React.FC<StatisticsTabProps> = ({ datasetId, statisticalAnalysis }) => {
  // Fetch AI summary using new AI service
  const { data: aiSummary, isLoading: aiSummaryLoading, error: aiSummaryError } = useQuery({
    queryKey: ['ai-summary', datasetId],
    queryFn: async () => {
      if (!datasetId) throw new Error('Dataset ID is required');
      console.log('Fetching AI summary for dataset:', datasetId);
      const result = await aiService.getStatisticalInsights(datasetId);
      console.log('AI Summary result:', result);
      return result;
    },
    enabled: !!datasetId,
    retry: 1,
  });

  return (
    <div>
      {/* AI Summary Section */}
      {aiSummaryLoading && (
        <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
          <div className="flex items-center">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-green-600 mr-2"></div>
            <span className="text-green-800">Loading AI statistical insights...</span>
          </div>
        </div>
      )}
      
      {aiSummaryError && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-800">Failed to load AI summary: {String(aiSummaryError)}</p>
        </div>
      )}
      
      {(() => {
        // Parse the AI insights data
        let summaryPoints = null;
        
        if (aiSummary?.data?.summary?.summary_points) {
          summaryPoints = aiSummary.data.summary.summary_points;
        } else if (aiSummary?.data?.insights) {
          try {
            // Extract JSON from the insights string
            const insightsStr = aiSummary.data.insights;
            const jsonMatch = insightsStr.match(/```json\n([\s\S]*?)\n```/);
            if (jsonMatch) {
              const parsedData = JSON.parse(jsonMatch[1]);
              summaryPoints = parsedData?.summary?.summary_points;
            }
          } catch (e) {
            console.error('Failed to parse AI insights:', e);
          }
        }
        
        return summaryPoints && (
          <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
            <h3 className="font-semibold mb-3 text-green-800 flex items-center">
              <span className="mr-2">🤖</span>
              AI Statistical Insights (10 Key Points)
            </h3>
            <div className="space-y-2">
              {summaryPoints.map((point: string, index: number) => (
                <div key={index} className="flex items-start">
                  <span className="inline-block w-6 h-6 bg-green-600 text-white text-xs rounded-full flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">
                    {index + 1}
                  </span>
                  <p className="text-green-900 text-sm">{point}</p>
                </div>
              ))}
            </div>
          </div>
        );
      })()}
      
      {/* Fallback: Show raw AI summary data if structure is different */}
      {aiSummary && !aiSummary?.data?.summary?.summary_points && (
        <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <h3 className="font-semibold mb-3 text-yellow-800 flex items-center">
            <span className="mr-2">🤖</span>
            AI Statistical Summary (Raw)
          </h3>
          <pre className="text-xs text-yellow-900 bg-yellow-100 p-2 rounded overflow-auto">
            {JSON.stringify(aiSummary, null, 2)}
          </pre>
        </div>
      )}
      
      <h3 className="font-semibold mb-4">Statistical Analysis</h3>
      
      {statisticalAnalysis ? (
        <div className="space-y-6">
          {/* Dataset Overview */}
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 p-6 rounded-lg border border-blue-200">
            <h4 className="font-semibold mb-4 text-lg text-gray-800">Dataset Overview</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-sm">
              <div className="space-y-1">
                <span className="font-medium text-gray-600">Shape:</span>
                <div className="text-lg font-semibold text-blue-600">
                  {statisticalAnalysis?.dataset_shape ? `${statisticalAnalysis.dataset_shape[0]?.toLocaleString()} × ${statisticalAnalysis.dataset_shape[1]}` : 'N/A'}
                </div>
              </div>
              <div className="space-y-1">
                <span className="font-medium text-gray-600">Total Columns:</span>
                <div className="text-lg font-semibold text-green-600">
                  {statisticalAnalysis?.dataset_overview?.total_columns || 0}
                </div>
              </div>
              <div className="space-y-1">
                <span className="font-medium text-gray-600">Numeric Columns:</span>
                <div className="text-lg font-semibold text-purple-600">
                  {statisticalAnalysis?.dataset_overview?.numeric_columns || 0}
                </div>
              </div>
              <div className="space-y-1">
                <span className="font-medium text-gray-600">Categorical Columns:</span>
                <div className="text-lg font-semibold text-orange-600">
                  {statisticalAnalysis?.dataset_overview?.categorical_columns || 0}
                </div>
              </div>
              <div className="space-y-1">
                <span className="font-medium text-gray-600">Memory Usage:</span>
                <div className="text-lg font-semibold text-indigo-600">
                  {statisticalAnalysis?.dataset_overview?.memory_usage_mb 
                    ? `${statisticalAnalysis.dataset_overview.memory_usage_mb.toFixed(2)} MB` 
                    : 'N/A'}
                </div>
              </div>
              <div className="space-y-1">
                <span className="font-medium text-gray-600">Missing Values:</span>
                <div className="text-lg font-semibold text-red-600">
                  {statisticalAnalysis?.dataset_overview?.missing_percentage !== undefined 
                    ? `${statisticalAnalysis.dataset_overview.missing_percentage.toFixed(2)}%` 
                    : '0%'}
                </div>
              </div>
              <div className="space-y-1">
                <span className="font-medium text-gray-600">Duplicate Rows:</span>
                <div className="text-lg font-semibold text-yellow-600">
                  {statisticalAnalysis?.dataset_overview?.duplicate_rows?.toLocaleString() || 0}
                </div>
              </div>
              <div className="space-y-1">
                <span className="font-medium text-gray-600">DateTime Columns:</span>
                <div className="text-lg font-semibold text-teal-600">
                  {statisticalAnalysis?.dataset_overview?.datetime_columns || 0}
                </div>
              </div>
            </div>
          </div>

          {/* Column Summary Table */}
          <div>
            <h4 className="font-medium mb-3">Column Summary</h4>
            <div className="overflow-x-auto">
              <table className="w-full text-sm border-collapse border border-gray-300">
                <thead>
                  <tr className="bg-gray-100">
                    <th className="border border-gray-300 px-4 py-3 text-left font-semibold min-w-[120px]">Column</th>
                    <th className="border border-gray-300 px-4 py-3 text-left font-semibold min-w-[100px]">Type</th>
                    <th className="border border-gray-300 px-4 py-3 text-left font-semibold min-w-[100px]">Data Type</th>
                    <th className="border border-gray-300 px-4 py-3 text-left font-semibold min-w-[90px]">Non-Null</th>
                    <th className="border border-gray-300 px-4 py-3 text-left font-semibold min-w-[90px]">Missing %</th>
                    <th className="border border-gray-300 px-4 py-3 text-left font-semibold min-w-[80px]">Unique</th>
                    <th className="border border-gray-300 px-4 py-3 text-left font-semibold min-w-[80px]">Min</th>
                    <th className="border border-gray-300 px-4 py-3 text-left font-semibold min-w-[80px]">Q1</th>
                    <th className="border border-gray-300 px-4 py-3 text-left font-semibold min-w-[80px]">Median</th>
                    <th className="border border-gray-300 px-4 py-3 text-left font-semibold min-w-[80px]">Q3</th>
                    <th className="border border-gray-300 px-4 py-3 text-left font-semibold min-w-[80px]">Max</th>
                    <th className="border border-gray-300 px-4 py-3 text-left font-semibold min-w-[80px]">Mean</th>
                    <th className="border border-gray-300 px-4 py-3 text-left font-semibold min-w-[90px]">Std Dev</th>
                  </tr>
                </thead>
                <tbody>
                  {statisticalAnalysis?.column_summary?.map((col: any, index: number) => (
                    <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                      <td className="border border-gray-300 px-4 py-3 font-medium">{col?.column_name || 'N/A'}</td>
                      <td className="border border-gray-300 px-4 py-3">
                        <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                          col?.column_type === 'numerical' 
                            ? 'bg-blue-100 text-blue-800' 
                            : 'bg-green-100 text-green-800'
                        }`}>
                          {col?.column_type || 'unknown'}
                        </span>
                      </td>
                      <td className="border border-gray-300 px-4 py-3">{col?.data_type || 'N/A'}</td>
                      <td className="border border-gray-300 px-4 py-3">{col?.non_null_count?.toLocaleString() || 'N/A'}</td>
                      <td className="border border-gray-300 px-4 py-3">
                        {col?.null_percentage !== undefined ? `${col.null_percentage.toFixed(1)}%` : 'N/A'}
                      </td>
                      <td className="border border-gray-300 px-4 py-3">{col?.unique_count?.toLocaleString() || 'N/A'}</td>
                      <td className="border border-gray-300 px-4 py-3">
                        {col?.column_type === 'numerical' && col?.min !== null && col?.min !== undefined 
                          ? Number(col.min).toFixed(3) : '-'}
                      </td>
                      <td className="border border-gray-300 px-4 py-3">
                        {col?.column_type === 'numerical' && col?.q25 !== null && col?.q25 !== undefined 
                          ? Number(col.q25).toFixed(3) : '-'}
                      </td>
                      <td className="border border-gray-300 px-4 py-3">
                        {col?.column_type === 'numerical' && col?.median !== null && col?.median !== undefined 
                          ? Number(col.median).toFixed(3) : '-'}
                      </td>
                      <td className="border border-gray-300 px-4 py-3">
                        {col?.column_type === 'numerical' && col?.q75 !== null && col?.q75 !== undefined 
                          ? Number(col.q75).toFixed(3) : '-'}
                      </td>
                      <td className="border border-gray-300 px-4 py-3">
                        {col?.column_type === 'numerical' && col?.max !== null && col?.max !== undefined 
                          ? Number(col.max).toFixed(3) : '-'}
                      </td>
                      <td className="border border-gray-300 px-4 py-3">
                        {col?.column_type === 'numerical' && col?.mean !== null && col?.mean !== undefined 
                          ? Number(col.mean).toFixed(3) : '-'}
                      </td>
                      <td className="border border-gray-300 px-4 py-3">
                        {col?.column_type === 'numerical' && col?.std !== null && col?.std !== undefined 
                          ? Number(col.std).toFixed(3) : '-'}
                      </td>
                    </tr>
                  )) || (
                    <tr>
                      <td colSpan={13} className="border border-gray-300 px-4 py-8 text-center text-gray-500">
                        No column data available
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Numerical Variables Analysis */}
          {statisticalAnalysis?.basic_statistics?.numeric_summary?.describe && 
           Object.keys(statisticalAnalysis.basic_statistics.numeric_summary.describe).length > 0 && (
            <div>
              <h4 className="font-medium mb-4">Numerical Variables Analysis</h4>
              <div className="space-y-4">
                {Object.entries(statisticalAnalysis.basic_statistics.numeric_summary.describe).map(([columnName, columnData]: [string, any]) => (
                  <NumericalVariableCard 
                    key={columnName} 
                    columnName={columnName} 
                    data={columnData} 
                  />
                ))}
              </div>
            </div>
          )}

          {/* Categorical Variables Analysis */}
          {statisticalAnalysis?.basic_statistics?.categorical_summary && 
           Object.keys(statisticalAnalysis.basic_statistics.categorical_summary).length > 0 && (
            <div>
              <h4 className="font-medium mb-4">Categorical Variables Analysis</h4>
              <div className="space-y-4">
                {Object.entries(statisticalAnalysis.basic_statistics.categorical_summary).map(([columnName, columnData]: [string, any]) => (
                  <CategoricalVariableCard 
                    key={columnName} 
                    columnName={columnName} 
                    data={columnData} 
                  />
                ))}
              </div>
            </div>
          )}

        </div>
      ) : (
        <div className="text-gray-500">Loading statistical analysis...</div>
      )}
    </div>
  );
};

export default StatisticsTab;
