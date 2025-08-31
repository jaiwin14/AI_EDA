#!/usr/bin/env python3
"""
Startup script for AI EDA
Launches both backend and frontend servers
"""

import subprocess
import sys
import os
import time
import threading
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

def start_frontend():
    """Start the frontend server"""
    print("🚀 Starting frontend server...")
    frontend_dir = Path(__file__).parent / "webchat-frontend"
    
    try:
        # Check if node_modules exists
        if not (frontend_dir / "node_modules").exists():
            print("📦 Installing frontend dependencies...")
            subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
        
        # Start development server
        subprocess.run([
            "npm", "run", "dev"
        ], cwd=frontend_dir, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Frontend server stopped")
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")

def main():
    """Main startup function"""
    print("🎯 AI EDA - Smart Exploratory Data Analysis")
    print("=" * 50)
    
    # Check if backend is ready
    backend_dir = Path(__file__).parent / "backend"
    if not (backend_dir / "requirements.txt").exists():
        print("❌ Backend directory not found!")
        return False
    
    # Check if frontend is ready
    frontend_dir = Path(__file__).parent / "webchat-frontend"
    if not (frontend_dir / "package.json").exists():
        print("❌ Frontend directory not found!")
        return False
    
    print("✅ Both backend and frontend directories found")
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # Wait a moment for backend to start
    time.sleep(3)
    
    # Start frontend
    start_frontend()
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down AI EDA...")
        sys.exit(0)
