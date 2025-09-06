from typing import Any, Dict, List, Union, Optional
from datetime import datetime, date, time
import numpy as np
import pandas as pd
import json

class JSONSerializer:
    @staticmethod
    def serialize_value(value: Any) -> Any:
        """Convert a single value to JSON serializable format with comprehensive handling."""
        if value is None:
            return None
        
        # Handle pandas and numpy dtype objects first - ENHANCED VERSION
        try:
            # Get type information
            type_name = str(type(value).__name__)
            module_name = getattr(type(value), '__module__', '')
            type_str = str(type(value))
            
            # Comprehensive dtype detection
            dtype_indicators = [
                'DType', 'Dtype', 'dtype', 'ObjectDType', 'CategoricalDtype',
                'IntervalDtype', 'PeriodDtype', 'DatetimeTZDtype', 'StringDtype',
                'Int64Dtype', 'Float64Dtype', 'BooleanDtype', 'ExtensionDtype'
            ]
            
            if (any(indicator in type_name for indicator in dtype_indicators) or
                'pandas.core.dtypes' in module_name or
                'pandas.core.arrays' in module_name or
                'numpy.dtypes' in module_name or
                'dtype' in type_str.lower()):
                return str(value)
                
            # Special handling for ObjectDType specifically
            if 'ObjectDType' in type_name or 'object' in str(value).lower():
                return str(value)
                
        except Exception:
            pass
        
        # Handle NaN/null values - ENHANCED
        try:
            # Multiple ways to check for NaN/null
            if pd.isna(value):
                return None
            if hasattr(value, 'isna') and callable(value.isna):
                if value.isna():
                    return None
            if hasattr(np, 'isnan') and not isinstance(value, (str, bool, type(None))):
                try:
                    if np.isnan(value):
                        return None
                except (TypeError, ValueError):
                    pass
        except (TypeError, ValueError, AttributeError):
            pass
        
        # Handle pandas-specific null types
        try:
            if hasattr(pd, 'NaType') and isinstance(value, type(pd.NaT)):
                return None
            if str(value) in ['NaT', 'NaN', '<NA>', 'None']:
                return None
        except Exception:
            pass
        
        # Handle numpy scalar types - ENHANCED
        if isinstance(value, np.generic):
            if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8,
                                 np.uint64, np.uint32, np.uint16, np.uint8)):
                return int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
                if np.isnan(value):
                    return None
                if np.isinf(value):
                    return str(value)
                return float(value)
            elif isinstance(value, (np.complexfloating, np.complex64, np.complex128)):
                if np.isnan(value):
                    return None
                return {"real": float(value.real), "imag": float(value.imag)}
            elif isinstance(value, np.bool_):
                return bool(value)
            else:
                # Fallback for other numpy scalars
                try:
                    return value.item()
                except (ValueError, AttributeError):
                    return str(value)
        
        # Handle Python native types
        if isinstance(value, (bool, int, float, str)):
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                return None if np.isnan(value) else str(value)
            return value
        
        # Handle date/time types - ENHANCED
        if isinstance(value, (datetime, date, time)):
            return value.isoformat()
        
        # Handle pandas timestamp types
        if isinstance(value, pd.Timestamp):
            if pd.isna(value):
                return None
            return value.isoformat()
        
        # Handle pandas Period, Interval, Timedelta
        if isinstance(value, (pd.Period, pd.Interval, pd.Timedelta)):
            if pd.isna(value):
                return None
            return str(value)
        
        # Handle pandas Index types
        if isinstance(value, (pd.Index, pd.MultiIndex)):
            return [JSONSerializer.serialize_value(v) for v in value.tolist()]
        
        # Handle pandas Series - ENHANCED
        if isinstance(value, pd.Series):
            try:
                return [JSONSerializer.serialize_value(v) for v in value.values]
            except Exception:
                # Fallback: convert to list and serialize each element
                return [JSONSerializer.serialize_value(v) for v in value.tolist()]
        
        # Handle pandas DataFrame - ENHANCED
        if isinstance(value, pd.DataFrame):
            try:
                return {
                    'data': [
                        {str(k): JSONSerializer.serialize_value(v) 
                         for k, v in row.items()}
                        for row in value.to_dict('records')
                    ],
                    'index': [JSONSerializer.serialize_value(idx) for idx in value.index],
                    'columns': [str(col) for col in value.columns]
                }
            except Exception:
                # Fallback: serialize to string representation
                return f"DataFrame(shape={value.shape})"
        
        # Handle pandas Categorical
        if hasattr(value, 'dtype') and str(value.dtype) == 'category':
            return JSONSerializer.serialize_value(value.astype(str))
        
        # Handle numpy arrays - ENHANCED
        if isinstance(value, np.ndarray):
            try:
                # Handle structured arrays
                if value.dtype.names is not None:
                    return [
                        {name: JSONSerializer.serialize_value(item[name]) for name in value.dtype.names}
                        for item in value
                    ]
                # Handle object arrays (common source of ObjectDType issues)
                elif value.dtype == np.dtype('O'):
                    return [JSONSerializer.serialize_value(item) for item in value.tolist()]
                # Handle regular arrays
                else:
                    return [JSONSerializer.serialize_value(v) for v in value.tolist()]
            except Exception:
                # Fallback for problematic arrays
                return f"ndarray(shape={value.shape}, dtype={value.dtype})"
        
        # Handle dictionaries - ENHANCED
        if isinstance(value, dict):
            try:
                return {str(k): JSONSerializer.serialize_value(v) for k, v in value.items()}
            except Exception:
                # If dict serialization fails, create a safe representation
                safe_dict = {}
                for k, v in value.items():
                    try:
                        safe_dict[str(k)] = JSONSerializer.serialize_value(v)
                    except Exception:
                        safe_dict[str(k)] = f"[Unserializable: {type(v).__name__}]"
                return safe_dict
        
        # Handle collections - ENHANCED
        if isinstance(value, (list, tuple, set, frozenset)):
            try:
                return [JSONSerializer.serialize_value(v) for v in value]
            except Exception:
                # Fallback for problematic collections
                safe_list = []
                for item in value:
                    try:
                        safe_list.append(JSONSerializer.serialize_value(item))
                    except Exception:
                        safe_list.append(f"[Unserializable: {type(item).__name__}]")
                return safe_list
        
        # Handle complex numbers
        if isinstance(value, complex):
            return {"real": float(value.real), "imag": float(value.imag)}
        
        # Handle pandas extension arrays and dtypes
        try:
            # Check for pandas extension types
            if hasattr(pd.api.types, 'is_extension_array_dtype'):
                if pd.api.types.is_extension_array_dtype(value):
                    return [JSONSerializer.serialize_value(v) for v in value]
            
            # Handle specific pandas dtypes
            if hasattr(value, '__module__') and 'pandas' in str(value.__module__):
                return str(value)
                
        except Exception:
            pass
        
        # Handle callable objects
        if callable(value):
            return f"[Callable: {getattr(value, '__name__', str(type(value).__name__))}]"
        
        # Handle objects with special string representations
        try:
            str_repr = str(value)
            # Check if the string representation is informative
            if str_repr and str_repr not in ['<object object at 0x', '[object Object]']:
                return str_repr
        except Exception:
            pass
        
        # Last resort: try to extract any serializable attributes
        try:
            if hasattr(value, '__dict__'):
                attrs = {k: JSONSerializer.serialize_value(v) for k, v in value.__dict__.items()
                        if not k.startswith('_')}
                if attrs:
                    return {"__type__": type(value).__name__, "attributes": attrs}
        except Exception:
            pass
        
        # Final fallback
        return f"[Object: {type(value).__name__}]"

    @staticmethod
    def convert_df_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame dtypes to JSON serializable formats with enhanced ObjectDType handling."""
        if df is None or df.empty:
            return df
        
        df = df.copy()
        
        try:
            for col in df.columns:
                try:
                    # Special handling for object dtype columns (ObjectDType source)
                    if df[col].dtype == 'object' or str(df[col].dtype) == 'object':
                        # Apply serialization to each value in object columns
                        df[col] = df[col].apply(JSONSerializer.serialize_value)
                    
                    # Handle datetime columns
                    elif pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = df[col].apply(lambda x: x.isoformat() if pd.notnull(x) else None)
                    
                    # Handle timedelta columns
                    elif pd.api.types.is_timedelta64_dtype(df[col]):
                        df[col] = df[col].astype(str)
                    
                    # Handle numeric columns
                    elif pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].apply(lambda x: JSONSerializer.serialize_value(x))
                    
                    # Handle categorical columns
                    elif pd.api.types.is_categorical_dtype(df[col]):
                        df[col] = df[col].astype(str).where(pd.notnull(df[col]), None)
                    
                    # Handle boolean columns
                    elif pd.api.types.is_bool_dtype(df[col]):
                        df[col] = df[col].apply(lambda x: bool(x) if pd.notnull(x) else None)
                    
                    # Handle complex columns
                    elif pd.api.types.is_complex_dtype(df[col]):
                        df[col] = df[col].apply(lambda x: {"real": float(x.real), "imag": float(x.imag)} if pd.notnull(x) else None)
                    
                    # Handle any other dtype
                    else:
                        df[col] = df[col].apply(JSONSerializer.serialize_value)
                
                except Exception as e:
                    # If individual column processing fails, apply serialization as fallback
                    print(f"Warning: Column '{col}' serialization failed ({e}), using fallback")
                    df[col] = df[col].apply(JSONSerializer.serialize_value)
        
        except Exception as e:
            print(f"Error in convert_df_dtypes: {e}")
            # Ultimate fallback: serialize everything
            for col in df.columns:
                try:
                    df[col] = df[col].apply(JSONSerializer.serialize_value)
                except Exception:
                    df[col] = df[col].astype(str).where(pd.notnull(df[col]), None)
        
        return df

    @staticmethod
    def test_serialization(obj: Any) -> bool:
        """Test if an object can be JSON serialized after processing."""
        try:
            serialized = JSONSerializer.serialize_value(obj)
            json.dumps(serialized)  # This will raise an exception if not serializable
            return True
        except Exception:
            return False

def to_json_serializable(obj: Any) -> Any:
    """Convert any Python/Pandas/Numpy object to JSON serializable format."""
    return JSONSerializer.serialize_value(obj)

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for JSON serialization with enhanced ObjectDType handling."""
    return JSONSerializer.convert_df_dtypes(df)

def create_dataframe_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Create a JSON-serializable summary of the DataFrame with enhanced error handling."""
    if df is None:
        return {"error": "DataFrame is None", "message": "Cannot generate summary for None"}
        
    if not isinstance(df, pd.DataFrame):
        return {"error": "Not a DataFrame", "message": f"Expected pandas DataFrame, got {type(df).__name__}"}
        
    try:
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Prepare the dataframe for serialization with enhanced handling
        df_prepared = prepare_dataframe(df_copy)
        
        # Basic information - with enhanced serialization
        info = {
            "shape": list(df.shape),
            "columns": [str(col) for col in df.columns],
            "dtypes": {str(k): JSONSerializer.serialize_value(v) for k, v in df.dtypes.items()},
            "memory_usage": int(df.memory_usage(deep=True).sum())
        }
        
        # Create a safe preview
        try:
            preview_data = []
            for _, row in df_prepared.head(5).iterrows():
                row_dict = {}
                for col in df_prepared.columns:
                    row_dict[str(col)] = JSONSerializer.serialize_value(row[col])
                preview_data.append(row_dict)
            info["preview"] = preview_data
        except Exception as e:
            info["preview_error"] = str(e)
            info["preview"] = "Preview generation failed"
        
        # Enhanced statistics
        statistics = {}
        try:
            if not df.empty:
                # Numeric statistics with better error handling
                try:
                    numeric_stats = df.describe().to_dict()
                    statistics["numeric"] = JSONSerializer.serialize_value(numeric_stats)
                except Exception:
                    statistics["numeric"] = "Statistics generation failed for numeric columns"
                
                # Categorical statistics with better error handling
                try:
                    cat_stats = df.describe(include=['object', 'category', 'bool']).to_dict()
                    statistics["categorical"] = JSONSerializer.serialize_value(cat_stats)
                except Exception:
                    statistics["categorical"] = "Statistics generation failed for categorical columns"
            else:
                statistics = {"numeric": {}, "categorical": {}}
        except Exception as e:
            statistics["error"] = str(e)
        
        # Missing value analysis with enhanced serialization
        missing_analysis = {}
        try:
            missing_analysis = {
                "total_missing": int(df.isnull().sum().sum()),
                "missing_by_column": JSONSerializer.serialize_value(df.isnull().sum().to_dict()),
                "missing_percentage": JSONSerializer.serialize_value((df.isnull().sum() / len(df) * 100).round(2).to_dict())
            }
        except Exception as e:
            missing_analysis["error"] = str(e)
        
        # Enhanced column type categorization
        column_types = {}
        try:
            column_types = {
                "numeric": [str(col) for col in df.select_dtypes(include=['number']).columns],
                "categorical": [str(col) for col in df.select_dtypes(include=['object', 'category']).columns],
                "datetime": [str(col) for col in df.select_dtypes(include=['datetime64']).columns],
                "boolean": [str(col) for col in df.select_dtypes(include=['bool']).columns],
                "timedelta": [str(col) for col in df.select_dtypes(include=['timedelta']).columns],
                "complex": [str(col) for col in df.select_dtypes(include=['complex']).columns]
            }
        except Exception as e:
            column_types["error"] = str(e)
        
        # Compile summary with complete serialization
        summary = {
            "info": info,
            "statistics": statistics,
            "missing_analysis": missing_analysis,
            "column_types": column_types,
            "generated_at": datetime.now().isoformat()
        }
        
        # Final serialization pass to ensure everything is JSON serializable
        return JSONSerializer.serialize_value(summary)
        
    except Exception as e:
        return {
            "error": str(e),
            "message": "Failed to generate summary",
            "error_type": type(e).__name__
        }

def serialize_nested_structure(data: Any) -> Any:
    """Recursively serialize nested data structures with comprehensive handling."""
    return JSONSerializer.serialize_value(data)

def infer_and_convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer and convert column types in a dataframe for better analysis.
    Enhanced version with better ObjectDType handling.
    """
    if df is None:
        return None
        
    df = df.copy()
    
    for column in df.columns:
        # Skip if column is already numeric and not object dtype
        if pd.api.types.is_numeric_dtype(df[column]) and df[column].dtype != 'object':
            continue
            
        # Special handling for object dtype columns
        if df[column].dtype == 'object':
            # First, try to clean up the column by handling mixed types
            try:
                # Check if column contains primarily strings that could be numbers
                sample = df[column].dropna().head(1000)
                if len(sample) > 0:
                    # Count how many look like numbers
                    numeric_count = 0
                    for val in sample:
                        try:
                            float(str(val).replace(',', ''))
                            numeric_count += 1
                        except:
                            pass
                    
                    # If most values look numeric, try conversion
                    if numeric_count / len(sample) > 0.7:
                        # Clean and convert to numeric
                        cleaned = df[column].astype(str).str.replace(',', '').str.strip()
                        numeric_series = pd.to_numeric(cleaned, errors='coerce')
                        if numeric_series.notna().sum() > 0.5 * len(numeric_series):
                            df[column] = numeric_series
                            continue
            except Exception:
                pass
        
        # Try converting to numeric if not already handled
        if not pd.api.types.is_numeric_dtype(df[column]):
            try:
                numeric_series = pd.to_numeric(df[column], errors='coerce')
                if numeric_series.notna().sum() > 0.5 * len(numeric_series):
                    df[column] = numeric_series
                    continue
            except Exception:
                pass
        
        # Try converting to datetime if not converted to numeric
        if not pd.api.types.is_numeric_dtype(df[column]):
            try:
                # Enhanced datetime conversion with better error handling
                sample = df[column].dropna().head(100)
                if len(sample) == 0:
                    continue
                
                # First check if values look like dates using regex patterns
                import re
                date_patterns = [
                                    (r'^\d{4}-\d{2}-\d{2}$', '%Y-%m-%d'),
                                    (r'^\d{4}/\d{2}/\d{2}$', '%Y/%m/%d'),
                                    (r'^\d{2}/\d{2}/\d{4}$', '%m/%d/%Y'),
                                    (r'^\d{2}-\d{2}-\d{4}$', '%m-%d-%Y'),
                                    (r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', '%Y-%m-%d %H:%M:%S'),
                                    (r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+$', '%Y-%m-%d %H:%M:%S.%f'),
                                    (r'^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}$', '%m/%d/%Y %H:%M:%S'),
                                ]
                    
            except Exception:
                pass
    
    return df

# Test function to verify serialization works
def test_serializer():
    """Test the serializer with various problematic types."""
    test_cases = [
        pd.DataFrame({'A': [1, 2, 3]}),
        np.array([1, 2, 3], dtype=object),
        pd.Series([1, 2, None], dtype=object),
        np.dtype('object'),
        pd.CategoricalDtype(['a', 'b', 'c']),
        {"complex": 1+2j, "nested": {"df": pd.DataFrame({'x': [1, 2]})}},
    ]
    
    for i, case in enumerate(test_cases):
        try:
            result = JSONSerializer.serialize_value(case)
            json.dumps(result)  # Test actual JSON serialization
            print(f"Test case {i+1}: PASSED")
        except Exception as e:
            print(f"Test case {i+1}: FAILED - {e}")
