#!/usr/bin/env python3
"""
Startup script for AI EDA Backend Only
Use this if you have issues with the frontend
"""

import subprocess
import sys
import os
from pathlib import Path

def start_backend():
    """Start the backend server"""
    print("🚀 Starting backend server...")
    backend_dir = Path(__file__).parent / "backend"
    
    try:
        # Change to backend directory and start server
        subprocess.run([
            sys.executable, "server.py"
        ], cwd=backend_dir, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")

def main():
    """Main startup function"""
    print("🎯 AI EDA - Backend Only")
    print("=" * 30)
    print("This will start only the backend server on http://localhost:8000")
    print("You can access the API documentation at http://localhost:8000/docs")
    print()
    
    # Check if backend is ready
    backend_dir = Path(__file__).parent / "backend"
    if not (backend_dir / "requirements.txt").exists():
        print("❌ Backend directory not found!")
        return False
    
    print("✅ Backend directory found")
    
    # Start backend
    start_backend()
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down AI EDA Backend...")
        sys.exit(0)
