#!/usr/bin/env python3
"""
Test script to check if the backend can start properly
"""

import sys
import os
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        # Test basic imports
        import pandas as pd
        print("✅ pandas imported successfully")
        
        import numpy as np
        print("✅ numpy imported successfully")
        
        import fastapi
        print("✅ fastapi imported successfully")
        
        # Test project-specific imports
        from utils.ai_utils import generate_insight, AIProvider
        print("✅ utils.ai_utils imported successfully")
        
        from utils.serialization_fixed import serialize_numpy, infer_and_convert_types, to_json_serializable
        print("✅ utils.serialization_fixed imported successfully")
        
        from backend.database import Database
        print("✅ backend.database imported successfully")
        
        # Test database initialization
        db = Database()
        print("✅ Database initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_environment():
    """Test environment variables"""
    print("\n🔍 Testing environment...")
    
    # Check if .env file exists
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        print("✅ .env file found")
        
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check for required API key
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key and gemini_key != "your_gemini_api_key_here":
            print("✅ GEMINI_API_KEY found")
        else:
            print("⚠️  GEMINI_API_KEY not set or using placeholder value")
    else:
        print("⚠️  .env file not found")
    
    return True

def main():
    """Main test function"""
    print("🧪 Testing AI EDA Backend")
    print("=" * 40)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test environment
    env_ok = test_environment()
    
    print("\n" + "=" * 40)
    if imports_ok and env_ok:
        print("✅ All tests passed! Backend should work properly.")
        print("\n🚀 You can now start the backend with:")
        print("   python start_backend_only.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\n💡 Make sure to:")
        print("   1. Install all dependencies: pip install -r backend/requirements.txt")
        print("   2. Set up your .env file with GEMINI_API_KEY")
        print("   3. Check that all import paths are correct")

if __name__ == "__main__":
    main()
