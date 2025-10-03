#!/usr/bin/env python3
"""
Test script to check all imports for missing values processor
"""

print("Testing imports...")

try:
    import pandas as pd
    print("✅ pandas imported successfully")
except ImportError as e:
    print(f"❌ pandas import failed: {e}")

try:
    import numpy as np
    print("✅ numpy imported successfully")
except ImportError as e:
    print(f"❌ numpy import failed: {e}")

try:
    from scipy import stats
    print("✅ scipy.stats imported successfully")
except ImportError as e:
    print(f"❌ scipy.stats import failed: {e}")

try:
    from sklearn.impute import SimpleImputer
    print("✅ SimpleImputer imported successfully")
except ImportError as e:
    print(f"❌ SimpleImputer import failed: {e}")

try:
    from sklearn.impute import KNNImputer
    print("✅ KNNImputer imported successfully")
except ImportError as e:
    print(f"❌ KNNImputer import failed: {e}")

try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    print("✅ IterativeImputer imported successfully")
except ImportError as e:
    print(f"❌ IterativeImputer import failed: {e}")

print("\nTesting app imports...")

try:
    from app.core.database import DatabaseManager, local_storage
    print("✅ database imports successful")
except ImportError as e:
    print(f"❌ database import failed: {e}")

try:
    from app.ml.ai_agent import ai_agent, AnalysisType
    print("✅ ai_agent imports successful")
except ImportError as e:
    print(f"❌ ai_agent import failed: {e}")

try:
    from app.ml.eda_processor import EDAProcessor
    print("✅ EDAProcessor import successful")
except ImportError as e:
    print(f"❌ EDAProcessor import failed: {e}")

try:
    from app.ml.missing_values_processor import MissingValuesProcessor
    print("✅ MissingValuesProcessor import successful")
except ImportError as e:
    print(f"❌ MissingValuesProcessor import failed: {e}")

print("\n✅ Import testing completed!")
