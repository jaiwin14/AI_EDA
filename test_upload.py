#!/usr/bin/env python3
"""
Test script to debug file upload issues
"""

import requests
import os
from pathlib import Path

def test_upload_endpoint():
    """Test the upload endpoint with a sample file"""
    
    # Backend URL
    base_url = "http://localhost:8000"
    
    # Test if backend is running
    try:
        response = requests.get(f"{base_url}/functions")
        if response.status_code == 200:
            print("✅ Backend is running")
        else:
            print("❌ Backend is not responding properly")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Backend is not running. Please start it with: python start_backend_only.py")
        return
    
    # Create a simple test CSV file
    test_csv = """name,age,city,date
John,25,New York,2023-01-15
Jane,30,Los Angeles,2023-02-20
Bob,35,Chicago,2023-03-10
Alice,28,Boston,2023-04-05"""
    
    # Save test file
    test_file_path = "test_data.csv"
    with open(test_file_path, "w") as f:
        f.write(test_csv)
    
    print(f"📁 Created test file: {test_file_path}")
    
    # Test upload
    try:
        with open(test_file_path, "rb") as f:
            files = {"file": ("test_data.csv", f, "text/csv")}
            response = requests.post(f"{base_url}/upload", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Upload successful!")
            print(f"   File ID: {result.get('file_id')}")
            print(f"   Filename: {result.get('filename')}")
            print(f"   Status: {result.get('status')}")
            
            # Test getting file summary
            file_id = result.get('file_id')
            if file_id:
                summary_response = requests.get(f"{base_url}/file/{file_id}/summary")
                if summary_response.status_code == 200:
                    print("✅ File summary retrieved successfully")
                else:
                    print(f"❌ Error getting file summary: {summary_response.status_code}")
                    print(f"   Response: {summary_response.text}")
        else:
            print(f"❌ Upload failed with status code: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during upload test: {str(e)}")
    
    # Clean up test file
    if os.path.exists(test_file_path):
        os.remove(test_file_path)
        print(f"🗑️  Cleaned up test file: {test_file_path}")

def test_environment():
    """Test environment configuration"""
    print("🔍 Testing environment...")
    
    # Check if .env file exists
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        print("✅ .env file found")
        
        # Load and check environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key and gemini_key != "your_gemini_api_key_here":
            print("✅ GEMINI_API_KEY found")
        else:
            print("⚠️  GEMINI_API_KEY not set or using placeholder value")
    else:
        print("⚠️  .env file not found")
    
    # Check if backend directory exists
    backend_dir = Path(__file__).parent / "backend"
    if backend_dir.exists():
        print("✅ Backend directory found")
    else:
        print("❌ Backend directory not found")

def main():
    """Main test function"""
    print("🧪 Testing AI EDA Upload Functionality")
    print("=" * 50)
    
    # Test environment
    test_environment()
    
    print("\n" + "=" * 50)
    
    # Test upload
    test_upload_endpoint()
    
    print("\n" + "=" * 50)
    print("🎯 Test completed!")

if __name__ == "__main__":
    main()
