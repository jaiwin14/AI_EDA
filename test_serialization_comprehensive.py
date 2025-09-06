import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.serialization_fixed import to_json_serializable

def test_serialization_comprehensive():
    print("\n=== Testing Comprehensive Serialization ===\n")
    
    # Create a DataFrame with various dtypes
    df = pd.DataFrame({
        'float64_col': [1.1, 2.2, 3.3, np.nan],
        'int64_col': [1, 2, 3, 4],
        'string_col': ['a', 'b', 'c', 'd'],
        'bool_col': [True, False, True, False],
        'datetime_col': pd.date_range('2023-01-01', periods=4),
        'category_col': pd.Series(['A', 'B', 'A', 'C']).astype('category')
    })
    
    # Add Float64 and Int64 columns if pandas version supports it
    try:
        df['Float64_col'] = pd.Series([1.1, 2.2, 3.3, None], dtype='Float64')
        print("Added Float64 column with dtype:", df['Float64_col'].dtype)
    except (TypeError, AttributeError):
        print("Pandas version doesn't support Float64 dtype")
        
    try:
        df['Int64_col'] = pd.Series([1, 2, 3, None], dtype='Int64')
        print("Added Int64 column with dtype:", df['Int64_col'].dtype)
    except (TypeError, AttributeError):
        print("Pandas version doesn't support Int64 dtype")
    
    # Test direct serialization of dtypes
    print("\n--- Testing Direct Dtype Serialization ---")
    for col in df.columns:
        dtype = df[col].dtype
        serialized = to_json_serializable(dtype)
        print(f"Column '{col}' dtype: {dtype} -> Serialized: {serialized}")
    
    # Test serialization in a dictionary
    print("\n--- Testing Dtype in Dictionary ---")
    dtype_dict = {col: df[col].dtype for col in df.columns}
    serialized_dict = to_json_serializable(dtype_dict)
    print(json.dumps(serialized_dict, indent=2))
    
    # Test serialization in a nested structure
    print("\n--- Testing Dtype in Nested Structure ---")
    nested_structure = {
        'dtypes': dtype_dict,
        'first_values': {col: df[col].iloc[0] for col in df.columns},
        'metadata': {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'index': df.index.tolist()
        }
    }
    serialized_nested = to_json_serializable(nested_structure)
    print(json.dumps(serialized_nested, indent=2))
    
    # Test serialization in an analysis-like structure
    print("\n--- Testing Analysis-like Structure ---")
    analysis_result = {
        'dataset_info': {
            'shape': df.shape,
            'dtypes': {col: df[col].dtype for col in df.columns},
            'memory_usage': df.memory_usage(deep=True).to_dict()
        },
        'column_stats': {
            col: {
                'dtype': df[col].dtype,
                'count': df[col].count(),
                'unique': len(df[col].unique()) if not pd.api.types.is_numeric_dtype(df[col]) else None,
                'mean': df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else None,
                'std': df[col].std() if pd.api.types.is_numeric_dtype(df[col]) else None,
                'min': df[col].min() if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_categorical_dtype(df[col]) else None,
                'max': df[col].max() if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_categorical_dtype(df[col]) else None
            } for col in df.columns
        }
    }
    serialized_analysis = to_json_serializable(analysis_result)
    
    # Verify JSON serializability
    try:
        json_str = json.dumps(serialized_analysis)
        print("Analysis result successfully serialized to JSON")
        # Print a sample of the JSON
        print(json_str[:500] + "..." if len(json_str) > 500 else json_str)
    except Exception as e:
        print(f"ERROR: Failed to serialize to JSON: {e}")
        raise

if __name__ == "__main__":
    test_serialization_comprehensive()
    print("\nAll tests completed successfully!")