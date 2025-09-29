import React from 'react';

interface StatisticsTableProps {
  columnName: string;
  data: number[];
  title?: string;
}

const StatisticsTable: React.FC<StatisticsTableProps> = ({ 
  columnName, 
  data, 
  title = "Summary Statistics" 
}) => {
  if (!data || data.length === 0) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
        <h6 className="font-medium text-gray-800 mb-2">{title}</h6>
        <div className="text-gray-500 text-center py-4">No data available</div>
      </div>
    );
  }

  // Calculate statistics
  const sortedData = [...data].sort((a, b) => a - b);
  const n = data.length;
  const mean = data.reduce((a, b) => a + b, 0) / n;
  const median = n % 2 === 0 
    ? (sortedData[n/2 - 1] + sortedData[n/2]) / 2 
    : sortedData[Math.floor(n/2)];
  
  const variance = data.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / (n - 1);
  const stdDev = Math.sqrt(variance);
  
  const q1 = sortedData[Math.floor(n * 0.25)];
  const q3 = sortedData[Math.floor(n * 0.75)];
  const iqr = q3 - q1;
  
  const min = sortedData[0];
  const max = sortedData[n - 1];
  
  // Calculate skewness
  const skewness = data.reduce((acc, val) => acc + Math.pow((val - mean) / stdDev, 3), 0) / n;
  
  // Calculate kurtosis
  const kurtosis = data.reduce((acc, val) => acc + Math.pow((val - mean) / stdDev, 4), 0) / n - 3;

  const statistics = [
    { label: 'Count', value: n },
    { label: 'Mean', value: mean.toFixed(3) },
    { label: 'Median', value: median.toFixed(3) },
    { label: 'Std Dev', value: stdDev.toFixed(3) },
    { label: 'Min', value: min.toFixed(3) },
    { label: 'Max', value: max.toFixed(3) },
    { label: 'Q1', value: q1.toFixed(3) },
    { label: 'Q3', value: q3.toFixed(3) },
    { label: 'IQR', value: iqr.toFixed(3) },
    { label: 'Skewness', value: skewness.toFixed(3) },
    { label: 'Kurtosis', value: kurtosis.toFixed(3) },
  ];

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
      <h6 className="font-medium text-gray-800 mb-3">{title} - {columnName}</h6>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-50">
              <th className="px-3 py-2 text-left font-medium text-gray-700">Statistic</th>
              <th className="px-3 py-2 text-left font-medium text-gray-700">Value</th>
            </tr>
          </thead>
          <tbody>
            {statistics.map((stat, index) => (
              <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                <td className="px-3 py-2 font-medium text-gray-600">{stat.label}</td>
                <td className="px-3 py-2 text-gray-800">{stat.value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default StatisticsTable;
