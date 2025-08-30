from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import uuid
from pathlib import Path
import pandas as pd
import importlib
import pkgutil
import sys
import os
from typing import Dict, Any
import asyncio

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.connection import (
    init_db,
    store_dataframe,
    load_dataframe,
    store_session,
    get_session_history
)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active connections
active_connections: Dict[str, WebSocket] = {}

# Dynamic EDA function loading
eda_functions = {}

def load_eda_functions():
    """Dynamically load all EDA functions from the eda directory"""
    eda_path = Path(__file__).parent / "eda"
    sys.path.append(str(eda_path.parent))
    
    for _, name, _ in pkgutil.iter_modules([str(eda_path)]):
        try:
            module = importlib.import_module(f"eda.{name}")
            if hasattr(module, 'run'):
                eda_functions[name] = module.run
        except ImportError as e:
            print(f"Error loading module {name}: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize database and load EDA functions"""
    init_db()
    load_eda_functions()

@app.get("/functions")
async def get_available_functions():
    """Return list of available EDA functions"""
    return {"functions": list(eda_functions.keys())}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file upload and store in database"""
    try:
        # Read the file
        content = await file.read()
        df = pd.read_csv(content)
        
        # Generate unique ID and store
        file_id = str(uuid.uuid4())
        store_dataframe(file_id, file.filename, df)
        
        return {"file_id": file_id, "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/status/{file_id}")
async def get_file_status(file_id: str):
    """Check if file exists and is ready for analysis"""
    df = load_dataframe(file_id)
    return {
        "exists": df is not None,
        "shape": df.shape if df is not None else None
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for interactive analysis"""
    await websocket.accept()
    client_id = str(uuid.uuid4())
    active_connections[client_id] = websocket
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["action"] == "run_analysis":
                function_name = data["function"]
                file_id = data["file_id"]
                
                # Load DataFrame
                df = load_dataframe(file_id)
                if df is None:
                    await websocket.send_json({
                        "type": "error",
                        "text": "File not found"
                    })
                    continue
                
                # Run analysis function
                if function_name in eda_functions:
                    try:
                        result = eda_functions[function_name](df)
                        
                        # Store session
                        session_id = str(uuid.uuid4())
                        store_session(session_id, json.dumps(data), result)
                        
                        # Send result
                        await websocket.send_json({
                            "type": "result",
                            "function": function_name,
                            **result
                        })
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "text": f"Error in analysis: {str(e)}"
                        })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "text": f"Function {function_name} not found"
                    })
                    
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if client_id in active_connections:
            del active_connections[client_id]

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
