"""
JSON Serialization Utilities
Handle numpy and pandas data types for JSON serialization
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict


def serialize_for_json(obj: Any) -> Any:
    """Convert numpy/pandas types to JSON-serializable types"""
    
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    else:
        return obj


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """Safely serialize object to JSON string"""
    serialized = serialize_for_json(obj)
    return json.dumps(serialized, **kwargs)


def clean_dataframe_for_json(df: pd.DataFrame) -> Dict[str, Any]:
    """Clean DataFrame for JSON serialization"""
    
    # Replace inf and -inf with None
    df_clean = df.replace([np.inf, -np.inf], None)
    
    # Convert to dict with proper types
    result = {}
    
    for column in df_clean.columns:
        series = df_clean[column]
        
        if series.dtype == 'object':
            # Handle object columns (strings, mixed types)
            result[column] = series.fillna(None).tolist()
        elif np.issubdtype(series.dtype, np.integer):
            # Handle integer columns
            result[column] = [int(x) if pd.notna(x) else None for x in series]
        elif np.issubdtype(series.dtype, np.floating):
            # Handle float columns
            result[column] = [float(x) if pd.notna(x) and not np.isinf(x) else None for x in series]
        else:
            # Handle other types
            result[column] = series.fillna(None).tolist()
    
    return result
