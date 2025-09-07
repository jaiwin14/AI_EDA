import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DatasetContext:
    """Container for dataset metadata and analysis results"""
    
    def __init__(self, df: pd.DataFrame, filename: str = ""):
        self.df = df
        self.filename = filename
        self.original_shape = df.shape
        self.schema_info = {}
        self.warnings = []
        self.target_column = None
        self.task_type = None
        
    def add_warning(self, message: str):
        self.warnings.append(message)
        logger.warning(message)

def safe_read_file(file_path: str, max_rows: Optional[int] = None) -> Tuple[pd.DataFrame, str]:
    """
    Safely read CSV/Excel/Parquet with error handling and sampling
    
    Args:
        file_path: Path to the file
        max_rows: Maximum rows to read (for sampling large files)
        
    Returns:
        Tuple of (DataFrame, error_message)
    """
    try:
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        
        # Read based on file extension
        if file_ext == '.csv':
            # Try different encodings and separators
            encodings = ['utf-8', 'latin-1', 'cp1252']
            separators = [',', ';', '\t', '|']
            
            for encoding in encodings:
                for sep in separators:
                    try:
                        df = pd.read_csv(
                            file_path, 
                            encoding=encoding, 
                            sep=sep,
                            nrows=max_rows,
                            low_memory=False
                        )
                        if df.shape[1] > 1:  # Valid if more than 1 column
                            return df, ""
                    except:
                        continue
                        
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, nrows=max_rows)
            return df, ""
            
        elif file_ext == '.parquet':
            df = pd.read_parquet(file_path)
            if max_rows:
                df = df.head(max_rows)
            return df, ""
            
        else:
            return pd.DataFrame(), f"Unsupported file format: {file_ext}"
            
    except Exception as e:
        return pd.DataFrame(), f"Error reading file: {str(e)}"
    
    return pd.DataFrame(), "Could not read file with any encoding/separator combination"

def infer_column_types(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Infer detailed column types and characteristics
    
    Returns:
        Dict with column info including type, cardinality, missing%, etc.
    """
    schema_info = {}
    
    for col in df.columns:
        col_info = {
            'dtype': str(df[col].dtype),
            'missing_count': df[col].isnull().sum(),
            'missing_pct': (df[col].isnull().sum() / len(df)) * 100,
            'unique_count': df[col].nunique(),
            'unique_pct': (df[col].nunique() / len(df)) * 100 if len(df) > 0 else 0,
            'is_constant': df[col].nunique() <= 1,
            'is_id_like': df[col].nunique() > 0.95 * len(df) and len(df) > 100,
        }
        
        # Infer semantic type
        if df[col].dtype in ['int64', 'float64']:
            if col_info['unique_count'] <= min(50, 0.2 * len(df)) and col_info['unique_count'] > 1:
                col_info['semantic_type'] = 'categorical_numeric'
            else:
                col_info['semantic_type'] = 'numeric'
        elif df[col].dtype == 'object':
            # Try to convert to datetime
            try:
                pd.to_datetime(df[col].dropna().head(100), errors='raise')
                col_info['semantic_type'] = 'datetime'
            except:
                if col_info['unique_count'] <= min(50, 0.2 * len(df)):
                    col_info['semantic_type'] = 'categorical'
                else:
                    col_info['semantic_type'] = 'text'
        else:
            col_info['semantic_type'] = 'other'
            
        schema_info[col] = col_info
    
    return schema_info

def get_sample_datasets():
    """Return sample datasets for testing"""
    from sklearn.datasets import load_iris, load_wine, load_diabetes, load_breast_cancer
    
    datasets = {}
    
    # Iris (classification)
    iris = load_iris(as_frame=True)
    iris_df = iris.frame
    datasets['Iris'] = iris_df
    
    # Wine (classification)
    wine = load_wine(as_frame=True)
    wine_df = wine.frame
    datasets['Wine'] = wine_df
    
    # Diabetes (regression)
    diabetes = load_diabetes(as_frame=True)
    diabetes_df = diabetes.frame
    datasets['Diabetes'] = diabetes_df
    
    # Breast Cancer (classification)
    cancer = load_breast_cancer(as_frame=True)
    cancer_df = cancer.frame
    datasets['Breast Cancer'] = cancer_df
    
    return datasets