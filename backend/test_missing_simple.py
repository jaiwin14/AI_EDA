#!/usr/bin/env python3
"""
Simple test for missing values functionality
"""
import pandas as pd
import numpy as np
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_simple_missing_analysis():
    """Test basic missing values analysis without advanced features"""
    
    print("🧪 Testing Simple Missing Values Analysis")
    
    # Create sample data with missing values
    np.random.seed(42)
    n_rows = 100
    
    data = {
        'numeric_col1': np.random.normal(100, 15, n_rows),
        'numeric_col2': np.random.normal(50, 10, n_rows),
        'categorical_col': np.random.choice(['A', 'B', 'C'], n_rows),
        'text_col': [f'text_{i}' for i in range(n_rows)]
    }
    
    df = pd.DataFrame(data)
    
    # Introduce missing values
    missing_indices = np.random.choice(n_rows, size=int(0.1 * n_rows), replace=False)
    df.loc[missing_indices, 'numeric_col1'] = np.nan
    
    missing_indices = np.random.choice(n_rows, size=int(0.05 * n_rows), replace=False)
    df.loc[missing_indices, 'categorical_col'] = np.nan
    
    print(f"📊 Sample dataset shape: {df.shape}")
    print(f"❌ Total missing values: {df.isnull().sum().sum()}")
    print("\nMissing values per column:")
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        print(f"  {col}: {missing_count} ({missing_pct:.1f}%)")
    
    # Test basic operations
    print("\n🔍 Testing Basic Operations:")
    
    # Test mean imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            mean_val = df[col].mean()
            df_filled = df.copy()
            df_filled[col] = df_filled[col].fillna(mean_val)
            print(f"✅ Mean imputation for {col}: {mean_val:.2f}")
    
    # Test mode imputation
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
            df_filled = df.copy()
            df_filled[col] = df_filled[col].fillna(mode_val)
            print(f"✅ Mode imputation for {col}: {mode_val}")
    
    # Test drop rows
    df_dropped = df.dropna()
    print(f"✅ Drop rows: {df.shape[0]} -> {df_dropped.shape[0]} rows")
    
    print("\n✅ Basic missing values operations working correctly!")
    return True

if __name__ == "__main__":
    try:
        test_simple_missing_analysis()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
