"""Backend server module with direct CSV to database handling."""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import pandas as pd
from datetime import datetime
from data_loader import DataLoader
from typing import Dict, Any, Optional

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data loader
data_loader = DataLoader()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Handle file upload and load directly into database.
    """
    try:
        # Create temporary file to store uploaded content
        temp_file_path = f"temp_{file.filename}"
        try:
            # Save uploaded file temporarily
            with open(temp_file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Load the CSV file into database
            result = data_loader.load_csv_to_db(temp_file_path)
            
            # Clean up temporary file
            os.remove(temp_file_path)
            
            return result
            
        finally:
            # Ensure temp file is removed even if an error occurs
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/{table_name}")
async def get_data(table_name: str, limit: int = 1000) -> Dict[str, Any]:
    """
    Retrieve data from a specific table.
    """
    try:
        data = data_loader.get_data_from_table(table_name, limit)
        return {
            'status': 'success',
            'data': data.to_dict(orient='records'),
            'columns': data.columns.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/table_info/{table_name}")
async def get_table_info(table_name: str) -> Dict[str, Any]:
    """
    Get information about a specific table.
    """
    try:
        return data_loader.get_table_info(table_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
