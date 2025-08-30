from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import json
import uuid
from pathlib import Path
import pandas as pd
import importlib
import pkgutil
import sys
import os
from typing import Dict, Any, Optional, List
import asyncio
import io

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our new database handler
from database import Database

app = FastAPI(title="AI EDA API", description="API for AI-powered Exploratory Data Analysis")

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

# Initialize database
db = Database()

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
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Load EDA functions
    load_eda_functions()

@app.get("/functions")
async def get_available_functions():
    """Return list of available EDA functions"""
    return {"functions": list(eda_functions.keys())}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Handle file upload and store in database"""
    try:
        # Read the file
        content = await file.read()
        
        # Determine file type and read accordingly
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or Excel file.")
        
        # Generate unique ID and store in database
        file_id = db.save_uploaded_file(df, file.filename)
        
        # Update file status to processing
        db.update_file_status(file_id, "processing")
        
        # Update file status to ready after processing
        if background_tasks:
            background_tasks.add_task(db.update_file_status, file_id, "ready")
        else:
            db.update_file_status(file_id, "ready")
        
        return {"file_id": file_id, "filename": file.filename, "status": "processing"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/status/{file_id}")
async def get_file_status(file_id: str):
    """Check if file exists and is ready for analysis"""
    file_info = db.get_file_status(file_id)
    
    if file_info.get("status") == "not_found":
        return JSONResponse(
            status_code=404,
            content={"exists": False, "message": "File not found"}
        )
    
    return {
        "exists": True,
        "status": file_info.get("status", "unknown"),
        "filename": file_info.get("original_filename", ""),
        "upload_time": file_info.get("upload_time", ""),
        "row_count": file_info.get("row_count", 0),
        "column_count": file_info.get("column_count", 0)
    }

@app.get("/files")
async def list_files():
    """List all uploaded files"""
    files = db.list_uploaded_files()
    return {"files": files}

@app.get("/download/{file_id}")
async def download_file(file_id: str, cleaned: bool = False):
    """Download a file as CSV"""
    try:
        # Get the dataframe
        df = db.get_dataframe(file_id, is_cleaned=cleaned)
        if df is None:
            return JSONResponse(
                status_code=404,
                content={"message": "File not found"}
            )
        
        # Create a temporary file
        temp_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', f"{file_id}.csv")
        df.to_csv(temp_file, index=False)
        
        # Get the original filename
        file_info = db.get_file_status(file_id)
        filename = file_info.get("original_filename", "download.csv")
        if cleaned:
            filename = f"cleaned_{filename}"
        
        return FileResponse(
            path=temp_file,
            filename=filename,
            media_type="text/csv"
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error downloading file: {str(e)}"}
        )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for interactive analysis"""
    await websocket.accept()
    client_id = str(uuid.uuid4())
    active_connections[client_id] = websocket
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "client_id": client_id
        })
        
        while True:
            data = await websocket.receive_json()
            
            # Handle both 'run_analysis' (new) and 'analyze' (old) actions
            if data["action"] == "run_analysis" or data["action"] == "analyze":
                # Extract function name from either 'function' or 'type' field
                function_name = data.get("function", data.get("type"))
                file_id = data.get("file_id")
                
                if not file_id:
                    await websocket.send_json({
                        "type": "error",
                        "text": "Missing file_id in request"
                    })
                    continue
                
                # Load DataFrame
                df = db.get_dataframe(file_id)
                if df is None:
                    await websocket.send_json({
                        "type": "error",
                        "text": "File not found"
                    })
                    continue
                
                # Run analysis function
                if function_name in eda_functions:
                    try:
                        # Send processing notification
                        await websocket.send_json({
                            "type": "processing",
                            "function": function_name,
                            "text": f"Processing {function_name} analysis..."
                        })
                        
                        # Handle clean_data function specially
                        if function_name == "clean_data":
                            # Get treatment options from request if available
                            treatment_options = data.get("treatment_options", None)
                            result = eda_functions[function_name](df, treatment_options)
                            
                            # If cleaning was successful and returned a file_id, update it in the response
                            if result.get("success") and result.get("file_id"):
                                result["cleaned_file_id"] = result.pop("file_id")
                        else:
                            # Run standard analysis function
                            result = eda_functions[function_name](df)
                        
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
            
            # Handle 'get_functions' action
            elif data["action"] == "get_functions":
                await websocket.send_json({
                    "type": "functions",
                    "functions": list(eda_functions.keys())
                })
            
            # Handle 'set_file' action for backward compatibility
            elif data["action"] == "set_file":
                await websocket.send_json({
                    "type": "info",
                    "text": "Please upload file through the /upload endpoint to get a file_id"
                })
                    
    except Exception as e:
        print(f"WebSocket error: {e}")
        # Try to send error message to client
        try:
            await websocket.send_json({
                "type": "connection_error",
                "text": f"Connection error: {str(e)}"
            })
        except:
            pass
    finally:
        if client_id in active_connections:
            del active_connections[client_id]

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
