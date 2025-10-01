import axios from 'axios';

export interface ColumnAnalysis {
  name: string;
  dtype: string;
  unique_values: number;
  missing_values: number;
  missing_percentage: number;
  is_numeric: boolean;
  is_categorical: boolean;
  is_datetime: boolean;
  is_binary: boolean;
  is_potential_target: boolean;
  potential_task_type?: string;
  sample_values: any[];
}

export interface DatasetAnalysis {
  filename: string;
  total_rows: number;
  total_columns: number;
  columns: ColumnAnalysis[];
  suggested_target?: string;
  suggested_task_type?: string;
  memory_usage_mb: number;
}

export const analyzeDataset = async (file: File): Promise<DatasetAnalysis> => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await axios.post('/api/v1/analyze', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  } catch (error) {
    console.error('Error analyzing dataset:', error);
    throw error;
  }
};
