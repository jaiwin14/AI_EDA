import sys
import os
import importlib

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    print("\n=== Testing All Imports ===\n")
    
    # Test utils.serialization_fixed imports
    print("Testing utils.serialization_fixed imports...")
    from utils.serialization_fixed import to_json_serializable, infer_and_convert_types
    print("✓ Successfully imported to_json_serializable and infer_and_convert_types from utils.serialization_fixed")
    
    # Test backend files that import from utils.serialization_fixed
    backend_files = [
        'backend.eda.bivariate_analysis',
        'backend.eda.analysis_summary',
        'backend.eda.dataset_info',
        'backend.eda.univariate_analysis',
        'backend.eda.outlier_detection_enhanced',
        'backend.eda.standardization_analysis',
        'backend.eda.encoding_analysis',
        'backend.eda.dimensionality_reduction'
        # Skipping these as they require database module
        # 'backend.routes.analytics',
        # 'backend.server'
    ]
    
    print("\nTesting backend files that import from utils.serialization_fixed...")
    for module_name in backend_files:
        try:
            module = importlib.import_module(module_name)
            print(f"✓ Successfully imported {module_name}")
        except ImportError as e:
            print(f"✗ Failed to import {module_name}: {str(e)}")
    
    # Test utils.data_utils imports
    print("\nTesting utils.data_utils imports...")
    try:
        from utils.data_utils import prepare_dataframe_summary
        print("✓ Successfully imported prepare_dataframe_summary from utils.data_utils")
    except ImportError as e:
        print(f"✗ Failed to import prepare_dataframe_summary from utils.data_utils: {str(e)}")
    
    print("\nAll import tests completed!")

if __name__ == "__main__":
    test_imports()