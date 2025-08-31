#!/usr/bin/env python3
"""
Setup script for AI EDA Backend
Installs dependencies and initializes the database
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing Python dependencies...")
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    requirements_file = script_dir / "requirements.txt"
    
    if not requirements_file.exists():
        print("Error: requirements.txt not found!")
        return False
    
    try:
        # Install requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def install_kaleido():
    """Install kaleido specifically for image export"""
    print("Installing kaleido for image export...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "kaleido>=0.2.1"
        ])
        print("✅ Kaleido installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing kaleido: {e}")
        return False

def test_kaleido():
    """Test if kaleido is working properly"""
    print("Testing kaleido installation...")
    
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        
        # Create a simple test plot
        fig = go.Figure(data=go.Bar(x=[1, 2, 3], y=[1, 3, 2]))
        
        # Try to export
        img_bytes = pio.to_image(fig, format="png", engine="kaleido")
        
        if img_bytes:
            print("✅ Kaleido is working correctly!")
            return True
        else:
            print("❌ Kaleido export returned empty result")
            return False
            
    except Exception as e:
        print(f"❌ Kaleido test failed: {e}")
        return False

def initialize_database():
    """Initialize the database"""
    print("Initializing database...")
    
    try:
        from database import Database
        
        # Initialize database
        db = Database()
        print("✅ Database initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        return False

def create_data_directory():
    """Create necessary directories"""
    print("Creating data directories...")
    
    try:
        script_dir = Path(__file__).parent
        data_dir = script_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        print(f"✅ Data directory created: {data_dir}")
        return True
    except Exception as e:
        print(f"❌ Error creating data directory: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up AI EDA Backend...")
    print("=" * 50)
    
    # Create data directory
    if not create_data_directory():
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Install kaleido specifically
    if not install_kaleido():
        return False
    
    # Test kaleido
    if not test_kaleido():
        print("⚠️  Kaleido test failed, but continuing setup...")
    
    # Initialize database
    if not initialize_database():
        return False
    
    print("=" * 50)
    print("✅ Setup completed successfully!")
    print("\nTo start the server, run:")
    print("python server.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
