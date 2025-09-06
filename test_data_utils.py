import pandas as pd
import numpy as np
from datetime import datetime
import json
from utils.data_utils import prepare_dataframe_summary

def test_data_utils():
    print("\n=== Testing Data Utils ===\n")
    
    # Create a test DataFrame with various data types
    df = pd.DataFrame({
        'float64_col': [1.1, 2.2, 3.3, np.nan],
        'int64_col': [1, 2, 3, 4],
        'string_col': ['a', 'b', 'c', 'd'],
        'bool_col': [True, False, True, False],
        'datetime_col': pd.date_range('2023-01-01', periods=4),
        'category_col': pd.Series(['A', 'B', 'A', 'C']).astype('category')
    })
    
    # Test prepare_dataframe_summary function
    print("Testing prepare_dataframe_summary function...")
    summary = prepare_dataframe_summary(df)
    
    # Verify the summary contains all expected sections
    expected_sections = ['info', 'describe', 'missing_analysis', 'column_types', 'generated_at']
    for section in expected_sections:
        if section in summary:
            print(f"✓ Section '{section}' found in summary")
        else:
            print(f"✗ Section '{section}' missing from summary")
    
    # Test JSON serialization
    print("\nTesting JSON serialization...")
    try:
        json_str = json.dumps(summary)
        print("✓ Summary successfully serialized to JSON")
        
        # Print a sample of the JSON
        print("\nSample of serialized JSON:")
        print(json_str[:500] + "..." if len(json_str) > 500 else json_str)
    except Exception as e:
        print(f"✗ ERROR: Failed to serialize to JSON: {e}")
        raise

if __name__ == "__main__":
    test_data_utils()
    print("\nAll tests completed successfully!")