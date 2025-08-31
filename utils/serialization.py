"""Utility functions for data serialization and type conversion."""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union

def serialize_numpy(obj: Any) -> Union[Dict, List, str, int, float, bool, None]:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif pd.isna(obj):
        return None
    return obj

def infer_and_convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """Intelligently infer and convert data types using AI-assisted pattern recognition."""
    df = df.copy()
    
    for column in df.columns:
        # Skip columns that are already numeric or datetime
        if pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_datetime64_dtype(df[column]):
            continue
            
        # Only process string/object columns
        if df[column].dtype == 'object':
            # Sample non-null values for type inference
            sample_values = df[column].dropna().sample(min(10, len(df[column].dropna()))).tolist()
            if not sample_values:
                continue
                
            # Try numeric conversion with various formats
            try:
                # Handle various number formats: "1,888", "1.888,00", "$1,888", etc.
                # First, try to clean the strings
                cleaned_values = df[column].astype(str).str.replace('$', '')
                cleaned_values = cleaned_values.str.replace('%', '')
                
                # Try European format (1.888,00) -> 1888.00
                if any(',' in str(val) and '.' in str(val) for val in sample_values):
                    # Check if it's likely European format
                    european_format = False
                    for val in sample_values:
                        if isinstance(val, str) and ',' in val and '.' in val:
                            if val.rindex(',') > val.rindex('.'):
                                european_format = True
                                break
                    
                    if european_format:
                        # Convert European format to US format
                        temp_values = cleaned_values.str.replace('.', '')
                        temp_values = temp_values.str.replace(',', '.')
                        if pd.to_numeric(temp_values, errors='coerce').notna().all():
                            df[column] = pd.to_numeric(temp_values, errors='coerce')
                            continue
                
                # Standard US format (1,888.00) -> 1888.00
                temp_values = cleaned_values.str.replace(',', '')
                if pd.to_numeric(temp_values, errors='coerce').notna().all():
                    df[column] = pd.to_numeric(temp_values, errors='coerce')
                    continue
            except (ValueError, AttributeError):
                pass
            
            # Try datetime conversion with multiple formats
            try:
                # Try common date formats first
                common_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d', '%m-%d-%Y', '%d-%m-%Y']
                converted = False
                
                for fmt in common_formats:
                    try:
                        temp_conversion = pd.to_datetime(df[column], format=fmt, errors='coerce')
                        if temp_conversion.notna().mean() > 0.8:  # More than 80% converted successfully
                            df[column] = temp_conversion
                            converted = True
                            break
                    except (ValueError, TypeError):
                        continue
                
                # If no specific format worked, try inferring
                if not converted:
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                    # Only keep the conversion if most values were successfully converted
                    if df[column].notna().mean() > 0.5:  # More than 50% converted successfully
                        continue
                    else:
                        # Revert back if conversion wasn't successful for most values
                        df[column] = df[column].astype('object')
            except (ValueError, TypeError):
                pass
            
            # Try boolean conversion for columns with few unique values
            if df[column].nunique() <= 2:
                try:
                    # Create a more comprehensive boolean mapping
                    bool_map = {
                        'true': True, 'false': False,
                        'yes': True, 'no': False,
                        'y': True, 'n': False,
                        't': True, 'f': False,
                        '1': True, '0': False,
                        1: True, 0: False,
                        'on': True, 'off': False,
                        'enable': True, 'disable': False,
                        'enabled': True, 'disabled': False
                    }
                    
                    # Case-insensitive mapping
                    lower_values = df[column].astype(str).str.lower()
                    if all(val in bool_map for val in lower_values.unique() if not pd.isna(val)):
                        df[column] = lower_values.map(bool_map)
                        continue
                except (ValueError, TypeError):
                    pass
    
    return df

def to_json_serializable(data: Any) -> Any:
    """Convert any data structure to JSON serializable format."""
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return serialize_numpy(data)
    elif isinstance(data, dict):
        return {k: to_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_json_serializable(v) for v in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.integer, np.floating, np.bool_)):
        return data.item()
    elif pd.isna(data):
        return None
    else:
        return serialize_numpy(data)
