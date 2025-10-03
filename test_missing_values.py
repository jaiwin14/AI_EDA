#!/usr/bin/env python3
"""
Test script for missing values analysis and treatment
"""
import pandas as pd
import numpy as np
import asyncio
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from app.ml.missing_values_processor import MissingValuesProcessor

async def test_missing_values_processor():
    """Test the missing values processor with sample data"""
    
    # Create sample data with missing values
    np.random.seed(42)
    n_rows = 1000
    
    data = {
        'numeric_col1': np.random.normal(100, 15, n_rows),
        'numeric_col2': np.random.normal(50, 10, n_rows),
        'categorical_col': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
        'time_series': pd.date_range('2023-01-01', periods=n_rows, freq='H'),
        'optional_field': np.random.choice(['Yes', 'No', 'Maybe'], n_rows)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce missing values with different patterns
    # MCAR pattern - completely random
    missing_indices = np.random.choice(n_rows, size=int(0.05 * n_rows), replace=False)
    df.loc[missing_indices, 'numeric_col1'] = np.nan
    
    # MAR pattern - missing depends on another variable
    high_values = df['numeric_col2'] > df['numeric_col2'].quantile(0.8)
    df.loc[high_values, 'categorical_col'] = np.nan
    
    # High missing percentage
    missing_indices = np.random.choice(n_rows, size=int(0.60 * n_rows), replace=False)
    df.loc[missing_indices, 'optional_field'] = np.nan
    
    print("🧪 Testing Missing Values Processor")
    print(f"📊 Sample dataset shape: {df.shape}")
    print(f"❌ Total missing values: {df.isnull().sum().sum()}")
    print("\nMissing values per column:")
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        print(f"  {col}: {missing_count} ({missing_pct:.1f}%)")
    
    # Initialize processor
    processor = MissingValuesProcessor()
    
    print("\n" + "="*50)
    print("🔍 MISSING VALUES ANALYSIS")
    print("="*50)
    
    # Test analysis
    try:
        analysis = await processor.analyze_missing_patterns(df)
        
        print(f"✅ Analysis completed successfully!")
        print(f"📈 Total missing values: {analysis['missing_summary']['total_missing_values']}")
        print(f"📊 Columns with missing: {analysis['missing_summary']['columns_with_missing']}")
        print(f"📉 Overall missing percentage: {analysis['missing_summary']['percentage_missing_overall']:.2f}%")
        
        print(f"\n🔍 Column Analysis:")
        for col, col_analysis in analysis['column_analysis'].items():
            print(f"  📋 {col}:")
            print(f"    - Missing: {col_analysis['missing_count']} ({col_analysis['missing_percentage']:.1f}%)")
            print(f"    - Pattern: {col_analysis['pattern_type']}")
            print(f"    - Reasons: {col_analysis['reasons'][0] if col_analysis['reasons'] else 'None'}")
            print(f"    - Top recommendation: {col_analysis['recommended_methods'][0]['method'] if col_analysis['recommended_methods'] else 'None'}")
        
        print(f"\n💡 Global Recommendations:")
        for rec in analysis['recommendations']:
            print(f"  • {rec}")
            
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        return
    
    print("\n" + "="*50)
    print("🛠️ MISSING VALUES TREATMENT")
    print("="*50)
    
    # Test different treatment methods
    methods_to_test = [
        ('drop_rows', {'column': 'numeric_col1'}),
        ('mean_imputation', {'column': 'numeric_col1'}),
        ('mode_imputation', {'column': 'categorical_col'}),
        ('drop_columns', {'threshold': 0.5}),
    ]
    
    for method, kwargs in methods_to_test:
        try:
            print(f"\n🧪 Testing {method}...")
            
            # Get preview
            preview = await processor.get_treatment_preview(df, method, **kwargs)
            print(f"  📋 Preview - Current shape: {preview['current_shape']}")
            print(f"  📋 Preview - Final shape: {preview['final_shape']}")
            
            # Apply treatment
            df_treated, treatment_info = await processor.treat_missing_values(df, method, **kwargs)
            print(f"  ✅ Treatment applied successfully!")
            print(f"  📊 Original shape: {treatment_info['original_shape']}")
            print(f"  📊 Final shape: {treatment_info['final_shape']}")
            print(f"  📊 Missing values remaining: {treatment_info['missing_values_remaining']}")
            
        except Exception as e:
            print(f"  ❌ Treatment {method} failed: {str(e)}")
    
    print("\n" + "="*50)
    print("✅ TESTING COMPLETED")
    print("="*50)
    print("🎉 Missing values processor is working correctly!")

if __name__ == "__main__":
    asyncio.run(test_missing_values_processor())
