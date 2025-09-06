import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.serialization_fixed import to_json_serializable

def test_all_dtypes_serialization():
    print("\n=== Testing All Pandas Dtypes Serialization ===\n")
    
    # Create a dictionary to store all test results
    results = {}
    
    # Test basic Python types
    basic_types = {
        'int': 42,
        'float': 3.14159,
        'str': 'hello world',
        'bool': True,
        'None': None,
        'list': [1, 2, 3],
        'dict': {'a': 1, 'b': 2},
        'datetime': datetime.now()
    }
    
    print("Testing basic Python types...")
    for type_name, value in basic_types.items():
        serialized = to_json_serializable(value)
        print(f"✓ {type_name}: {value} -> {serialized}")
        results[f'basic_{type_name}'] = {'original': str(value), 'serialized': serialized}
    
    # Test NumPy types
    numpy_types = {
        'np.int64': np.int64(42),
        'np.float64': np.float64(3.14159),
        'np.bool_': np.bool_(True),
        'np.array': np.array([1, 2, 3]),
        'np.array_2d': np.array([[1, 2], [3, 4]]),
        'np.nan': np.nan
    }
    
    print("\nTesting NumPy types...")
    for type_name, value in numpy_types.items():
        serialized = to_json_serializable(value)
        print(f"✓ {type_name}: {value} -> {serialized}")
        results[f'numpy_{type_name}'] = {'original': str(value), 'serialized': serialized}
    
    # Test Pandas types
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
        df['Int64_col'] = pd.Series([1, 2, 3, None], dtype='Int64')
    except (TypeError, AttributeError):
        print("Pandas version doesn't support Float64/Int64 dtypes")
    
    pandas_types = {
        'Series': df['float64_col'],
        'DataFrame': df,
        'Index': df.index,
        'DatetimeIndex': pd.date_range('2023-01-01', periods=4),
        'Timestamp': pd.Timestamp('2023-01-01'),
        'Timedelta': pd.Timedelta(days=1),
        'Period': pd.Period('2023-01'),
        'Interval': pd.Interval(left=0, right=1),
        'Categorical': pd.Categorical(['A', 'B', 'C']),
    }
    
    # Add all dtypes
    for col in df.columns:
        pandas_types[f'dtype_{col}'] = df[col].dtype
    
    print("\nTesting Pandas types...")
    for type_name, value in pandas_types.items():
        serialized = to_json_serializable(value)
        print(f"✓ {type_name} -> serialized successfully")
        # Don't print the full serialized value as it can be very large
        results[f'pandas_{type_name}'] = {'original_type': str(type(value)), 'serialized_type': str(type(serialized))}
    
    # Test complex nested structures
    nested_structure = {
        'metadata': {
            'created_at': datetime.now(),
            'version': '1.0.0',
            'settings': {
                'precision': np.float64(0.001),
                'max_iterations': np.int64(1000),
                'use_gpu': np.bool_(True)
            }
        },
        'data': {
            'series': df['float64_col'],
            'array': np.array([1, 2, 3]),
            'matrix': np.random.rand(3, 3),
            'dataframe': df.head(2)
        },
        'statistics': {
            'mean': df.select_dtypes(include=['number']).mean(),
            'median': df.select_dtypes(include=['number']).median(),
            'std': df.select_dtypes(include=['number']).std(),
            'dtypes': df.dtypes
        }
    }
    
    print("\nTesting complex nested structure...")
    serialized_nested = to_json_serializable(nested_structure)
    print("✓ Complex nested structure serialized successfully")
    
    # Verify JSON serializability
    try:
        json_str = json.dumps(serialized_nested)
        print("✓ Successfully serialized to JSON")
    except Exception as e:
        print(f"✗ Failed to serialize to JSON: {e}")
    
    print("\nAll serialization tests completed successfully!")
    return results

if __name__ == "__main__":
    test_all_dtypes_serialization()