#!/usr/bin/env python3
"""
Test script to verify AI EDA setup
"""

import sys
import os
import pandas as pd
import numpy as np

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import fastapi
        print("✅ FastAPI imported successfully")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    try:
        import kaleido
        print("✅ Kaleido imported successfully")
    except ImportError as e:
        print(f"❌ Kaleido import failed: {e}")
        return False
    
    try:
        import pandas as pd
        import numpy as np
        print("✅ Pandas and NumPy imported successfully")
    except ImportError as e:
        print(f"❌ Pandas/NumPy import failed: {e}")
        return False
    
    return True

def test_kaleido():
    """Test kaleido image export"""
    print("\nTesting kaleido image export...")
    
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        
        # Create a simple test plot
        fig = go.Figure(data=go.Bar(x=[1, 2, 3], y=[1, 3, 2]))
        
        # Try to export
        img_bytes = pio.to_image(fig, format="png", engine="kaleido")
        
        if img_bytes and len(img_bytes) > 0:
            print("✅ Kaleido image export working correctly")
            return True
        else:
            print("❌ Kaleido export returned empty result")
            return False
            
    except Exception as e:
        print(f"❌ Kaleido test failed: {e}")
        return False

def test_database():
    """Test database functionality"""
    print("\nTesting database...")
    
    try:
        from database import Database
        
        # Initialize database
        db = Database()
        print("✅ Database initialized successfully")
        
        # Create test data
        test_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 4, 6, 8, 10],
            'C': ['a', 'b', 'c', 'd', 'e']
        })
        
        # Save test data
        file_id = db.save_uploaded_file(test_df, "test.csv")
        print(f"✅ Test data saved with ID: {file_id}")
        
        # Retrieve test data
        retrieved_df = db.get_dataframe(file_id)
        if retrieved_df is not None and len(retrieved_df) == len(test_df):
            print("✅ Test data retrieved successfully")
        else:
            print("❌ Test data retrieval failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_eda_modules():
    """Test EDA modules"""
    print("\nTesting EDA modules...")
    
    # Create test data
    test_df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'B': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
        'D': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    })
    
    # Add some missing values for testing
    test_df.loc[2, 'A'] = np.nan
    test_df.loc[5, 'B'] = np.nan
    
    modules_to_test = [
        'summary',
        'correlation', 
        'data_distribution',
        'missing_values',
        'outlier_detection'
    ]
    
    success_count = 0
    
    for module_name in modules_to_test:
        try:
            # Import module
            module = __import__(f'eda.{module_name}', fromlist=['run'])
            
            # Run analysis
            result = module.run(test_df)
            
            # Check if result has expected structure
            if isinstance(result, dict) and 'text' in result:
                print(f"✅ {module_name}: OK")
                success_count += 1
            else:
                print(f"❌ {module_name}: Invalid result structure")
                
        except Exception as e:
            print(f"❌ {module_name}: ERROR - {e}")
    
    print(f"\nEDA modules test: {success_count}/{len(modules_to_test)} passed")
    return success_count == len(modules_to_test)

def test_utils():
    """Test utility functions"""
    print("\nTesting utility functions...")
    
    try:
        from eda.utils import export_plot_as_image, create_error_plot
        
        # Test plot export
        import plotly.graph_objects as go
        fig = go.Figure(data=go.Bar(x=[1, 2, 3], y=[1, 3, 2]))
        
        # Test normal export
        result = export_plot_as_image(fig)
        if result:
            print("✅ Plot export utility working")
        else:
            print("❌ Plot export utility failed")
            return False
        
        # Test error plot
        error_plot = create_error_plot("Test error message")
        if error_plot:
            print("✅ Error plot utility working")
        else:
            print("❌ Error plot utility failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Utility functions test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 AI EDA Setup Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Kaleido", test_kaleido),
        ("Database", test_database),
        ("EDA Modules", test_eda_modules),
        ("Utilities", test_utils)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your AI EDA setup is working correctly.")
        print("\nYou can now start the server with:")
        print("python server.py")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Install kaleido: pip install kaleido>=0.2.1")
        print("3. Check database permissions")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
