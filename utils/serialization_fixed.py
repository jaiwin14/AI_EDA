"""Utility functions for data serialization and type conversion."""
from typing import Any, Dict, List, Union, Optional
from datetime import datetime, date, time
import numpy as np
import pandas as pd

class JSONSerializer:
    @staticmethod
    def serialize_value(value: Any) -> Any:
        """Convert a single value to JSON serializable format."""
        if value is None:
            return None
            
        if pd.isna(value) or pd.isnull(value):
            return None
            
        if isinstance(value, (np.integer, np.int64)):
            return int(value)
            
        if isinstance(value, (np.floating, np.float64)):
            if np.isnan(value):
                return None
            if np.isinf(value):
                return str(value)
            return float(value)
            
        if isinstance(value, (datetime, date, time, pd.Timestamp)):
            return value.isoformat()
            
        if isinstance(value, (pd.Period, pd.Interval, pd.Timedelta)):
            return str(value)
            
        if isinstance(value, np.bool_):
            return bool(value)
            
        if isinstance(value, (pd.Index, pd.MultiIndex)):
            return list(map(str, value.tolist()))
            
        if isinstance(value, pd.Series):
            return [JSONSerializer.serialize_value(v) for v in value]
            
        if isinstance(value, pd.DataFrame):
            return {
                'data': [
                    {str(k): JSONSerializer.serialize_value(v) 
                     for k, v in row.items()}
                    for row in value.to_dict('records')
                ],
                'index': list(map(str, value.index)),
                'columns': list(map(str, value.columns))
            }
            
        if isinstance(value, np.ndarray):
            return [JSONSerializer.serialize_value(v) for v in value.tolist()]
            
        if isinstance(value, dict):
            return {str(k): JSONSerializer.serialize_value(v) 
                   for k, v in value.items()}
            
        if isinstance(value, (list, tuple, set)):
            return [JSONSerializer.serialize_value(v) for v in value]
            
        return str(value)

    @staticmethod
    def convert_df_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame dtypes to JSON serializable formats."""
        df = df.copy()
        
        # Convert datetime columns to ISO format strings
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            df[col] = df[col].apply(lambda x: x.isoformat() if pd.notnull(x) else None)
        
        # Convert timedelta to strings
        timedelta_cols = df.select_dtypes(include=['timedelta64']).columns
        for col in timedelta_cols:
            df[col] = df[col].astype(str)
        
        # Convert numeric to native Python types
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            df[col] = df[col].astype(float).where(pd.notnull(df[col]), None)
        
        # Convert categorical to strings
        categorical_cols = df.select_dtypes(include=['category']).columns
        for col in categorical_cols:
            df[col] = df[col].astype(str).where(pd.notnull(df[col]), None)
        
        return df

def to_json_serializable(obj: Any) -> Any:
    """Convert any Python/Pandas/Numpy object to JSON serializable format."""
    return JSONSerializer.serialize_value(obj)

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for JSON serialization."""
    return JSONSerializer.convert_df_dtypes(df)

def create_dataframe_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Create a JSON-serializable summary of the DataFrame."""
    try:
        df = prepare_dataframe(df)
        
        summary = {
            "info": {
                "shape": list(df.shape),
                "columns": list(map(str, df.columns)),
                "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
                "missing_values": df.isnull().sum().to_dict(),
                "memory_usage": int(df.memory_usage(deep=True).sum()),
                "preview": df.head(5).to_dict('records')
            },
            "statistics": {
                "numeric": df.describe().to_dict() if not df.empty else {},
                "categorical": df.describe(include=['object', 'category']).to_dict() if not df.empty else {}
            },
            "missing_analysis": {
                "total_missing": int(df.isnull().sum().sum()),
                "missing_by_column": df.isnull().sum().to_dict(),
                "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict()
            },
            "column_types": {
                "numeric": df.select_dtypes(include=['number']).columns.tolist(),
                "categorical": df.select_dtypes(include=['object', 'category']).columns.tolist(),
                "datetime": df.select_dtypes(include=['datetime64']).columns.tolist(),
                "boolean": df.select_dtypes(include=['bool']).columns.tolist()
            }
        }
        
        return JSONSerializer.serialize_value(summary)
    except Exception as e:
        return {
            "error": str(e),
            "message": "Failed to generate summary"
        }

def infer_and_convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer and convert column types in a dataframe for better analysis.
    Returns a new DataFrame with converted types.
    """
    df = df.copy()
    
    for column in df.columns:
        # Skip if column is already numeric
        if pd.api.types.is_numeric_dtype(df[column]):
            continue
            
        # Try converting to numeric
        try:
            numeric_series = pd.to_numeric(df[column], errors='coerce')
            # If successful and didn't create too many NaN values, convert
            if numeric_series.notna().sum() > 0.5 * len(numeric_series):
                df[column] = numeric_series
        except:
            pass
            
        # Try converting to datetime if not converted to numeric
        if not pd.api.types.is_numeric_dtype(df[column]):
            try:
                # Try common date formats first
                formats = [
                    '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', 
                    '%Y/%m/%d', '%d-%m-%Y', '%m-%d-%Y',
                    '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f',
                    '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S'
                ]
                
                # First try to detect the format from a sample
                sample = df[column].dropna().head(100)  # Sample up to 100 non-null values
                best_format = None
                max_success = 0
                
                for fmt in formats:
                    try:
                        success = sum(1 for val in sample 
                                    if pd.to_datetime(str(val), format=fmt, errors='coerce') is not pd.NaT)
                        if success > max_success:
                            max_success = success
                            best_format = fmt
                    except:
                        continue
                
                # If we found a good format that works for most values, use it
                if best_format and max_success / len(sample) > 0.8:
                    datetime_series = pd.to_datetime(df[column], format=best_format, errors='coerce')
                    if datetime_series.notna().mean() > 0.8:  # If 80% or more are valid
                        df[column] = datetime_series
                else:
                    # If no specific format worked well, try some common patterns first
                    patterns = [
                        r'^\d{4}-\d{2}-\d{2}',  # ISO date
                        r'^\d{2}/\d{2}/\d{4}',   # MM/DD/YYYY or DD/MM/YYYY
                        r'^\d{2}-\d{2}-\d{4}',   # MM-DD-YYYY or DD-MM-YYYY
                    ]
                    
                    # Try to identify the pattern from the sample
                    import re
                    pattern_matches = {pattern: sum(1 for val in sample if re.match(pattern, str(val)))
                                    for pattern in patterns}
                    
                    best_pattern = max(pattern_matches.items(), key=lambda x: x[1])[0] if pattern_matches else None
                    
                    if best_pattern and pattern_matches[best_pattern] / len(sample) > 0.8:
                        # Use the identified pattern to guide parsing
                        if best_pattern == r'^\d{4}-\d{2}-\d{2}':
                            datetime_series = pd.to_datetime(df[column], format='%Y-%m-%d', errors='coerce')
                        elif best_pattern in [r'^\d{2}/\d{2}/\d{4}', r'^\d{2}-\d{2}-\d{4}']:
                            # Try both US and UK formats
                            us_format = pd.to_datetime(df[column], format='%m/%d/%Y', errors='coerce')
                            uk_format = pd.to_datetime(df[column], format='%d/%m/%Y', errors='coerce')
                            # Use the one that parsed more successfully
                            datetime_series = us_format if us_format.notna().sum() > uk_format.notna().sum() else uk_format
                    else:
                        # As a last resort, use the flexible parser
                        datetime_series = pd.to_datetime(df[column], errors='coerce')
                    
                    if datetime_series.notna().mean() > 0.8:
                        df[column] = datetime_series
            except:
                pass
    
    return df
