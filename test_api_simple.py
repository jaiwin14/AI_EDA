#!/usr/bin/env python3
"""
Simple API test for missing values endpoint
"""
import requests
import json
import pandas as pd
import numpy as np

def create_test_dataset():
    """Create a test dataset with missing values"""
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
    
    return df

def test_missing_values_endpoint():
    """Test the missing values analysis endpoint"""
    
    print("🧪 Testing Missing Values API Endpoint")
    
    # Test dataset ID (replace with actual dataset ID)
    dataset_id = "test_dataset_123"
    
    # Test the endpoint
    url = f"http://localhost:8000/api/v1/ai-insights/{dataset_id}/missing-values"
    
    print(f"📡 Testing URL: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Request successful!")
            print(f"📊 Response structure:")
            
            if 'data' in data:
                analysis = data['data']
                print(f"  - Total missing values: {analysis.get('missing_summary', {}).get('total_missing_values', 'N/A')}")
                print(f"  - Columns with missing: {analysis.get('missing_summary', {}).get('columns_with_missing', 'N/A')}")
                print(f"  - Overall missing %: {analysis.get('missing_summary', {}).get('percentage_missing_overall', 'N/A'):.2f}%")
                
                if 'column_analysis' in analysis:
                    print(f"  - Columns analyzed: {len(analysis['column_analysis'])}")
                    for col, col_data in list(analysis['column_analysis'].items())[:3]:
                        print(f"    * {col}: {col_data.get('missing_percentage', 0):.1f}% missing")
                
                if 'recommendations' in analysis:
                    print(f"  - Recommendations: {len(analysis['recommendations'])}")
                    for rec in analysis['recommendations'][:2]:
                        print(f"    * {rec}")
            
            return True
            
        elif response.status_code == 404:
            print(f"❌ Dataset not found (404). Make sure dataset {dataset_id} exists.")
            return False
            
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure the backend is running on port 8000")
        return False
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_with_sample_data():
    """Test with sample data structure"""
    
    print("\n🧪 Testing with Sample Data Structure")
    
    # Create sample data
    df = create_test_dataset()
    
    print(f"📊 Created test dataset: {df.shape}")
    print(f"❌ Missing values: {df.isnull().sum().sum()}")
    
    # Simulate what the analysis would return
    missing_summary = {
        "total_missing_values": int(df.isnull().sum().sum()),
        "columns_with_missing": int((df.isnull().sum() > 0).sum()),
        "percentage_missing_overall": float(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
    }
    
    print(f"✅ Analysis would show:")
    print(f"  - Total missing: {missing_summary['total_missing_values']}")
    print(f"  - Columns affected: {missing_summary['columns_with_missing']}")
    print(f"  - Overall %: {missing_summary['percentage_missing_overall']:.2f}%")
    
    return True

if __name__ == "__main__":
    print("🚀 Starting API Tests\n")
    
    # Test with sample data first
    test_with_sample_data()
    
    # Test the actual endpoint
    print("\n" + "="*50)
    endpoint_result = test_missing_values_endpoint()
    
    if endpoint_result:
        print("\n🎉 API endpoint test passed!")
    else:
        print("\n⚠️  API endpoint test failed, but this might be expected if:")
        print("   - Backend server is not running")
        print("   - Dataset doesn't exist")
        print("   - There are import/dependency issues")
        
    print("\n✅ Test completed!")
