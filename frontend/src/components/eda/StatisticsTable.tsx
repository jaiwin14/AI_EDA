import React from 'react';
import './StatisticsTable.css';

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
      <div className="stats-table-card">
        <h6 className="stats-table-title">{title}</h6>
        <div className="stats-table-empty">No data available</div>
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
    <div className="stats-table-card">
      <h6 className="stats-table-title">{title} - {columnName}</h6>
      <div className="stats-table-container">
        <table className="stats-table">
          <thead>
            <tr>
              <th>Statistic</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            {statistics.map((stat, index) => (
              <tr key={index}>
                <td className="stats-table-stat-label">{stat.label}</td>
                <td className="stats-table-stat-value">{stat.value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default StatisticsTable;
