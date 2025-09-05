import sys
import os
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import json
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
import importlib
import pkgutil
from typing import Dict, Any, Optional, List
import asyncio
import io
from dotenv import load_dotenv
from utils.serialization import infer_and_convert_types, to_json_serializable

# Load environment variables
load_dotenv()



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

# Store file summaries in memory (you might want to move this to a proper database)
file_summaries: Dict[str, Dict] = {}

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

@app.get("/file/{file_id}/summary")
async def get_file_summary(file_id: str):
    """Get summary statistics and information about a file"""
    try:
        summary = db.get_file_summary(file_id)
        if not summary:
            raise HTTPException(status_code=404, detail="File not found")
        return JSONResponse(content=summary)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/file/{file_id}/missing")
async def analyze_missing_values(file_id: str):
    """Analyze missing values in the dataset"""
    try:
        df = db.get_dataframe(file_id)
        if df is None:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Calculate missing value statistics
        missing_stats = {
            "total_missing": df.isnull().sum().sum(),
            "missing_by_column": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "missing_pattern": df.isnull().sum(axis=1).value_counts().to_dict(),
            "missing_correlation": df.isnull().corr().to_dict()
        }
        
        # Generate AI insights about missing values
        insight_prompt = f"""
        Analyze the missing value patterns in this dataset:
        - Total missing values: {missing_stats['total_missing']}
        - Missing percentages by column: {missing_stats['missing_percentage']}
        - Missing value patterns: {missing_stats['missing_pattern']}
        
        Please provide:
        1. Assessment of missing data patterns (MCAR, MAR, or MNAR)
        2. Recommendations for handling missing values
        3. Potential impact on analysis
        """
        
        try:
            missing_stats["ai_insights"] = generate_insight(insight_prompt, context=missing_stats)
        except:
            missing_stats["ai_insights"] = "Unable to generate AI insights at this time."
        
        return JSONResponse(content=to_json_serializable(missing_stats))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/file/{file_id}/missing/treat")
async def treat_missing_values(
    file_id: str,
    treatment_method: str = "mean",  # mean, median, mode, drop
    columns: List[str] = None  # If None, treat all columns
):
    """Treat missing values in the dataset"""
    try:
        df = db.get_dataframe(file_id)
        if df is None:
            raise HTTPException(status_code=404, detail="File not found")
        
        df_treated = df.copy()
        columns_to_treat = columns or df.columns
        
        # Treatment statistics
        treatment_stats = {"treated_columns": {}}
        
        for col in columns_to_treat:
            if col not in df.columns:
                continue
                
            missing_count = df[col].isnull().sum()
            if missing_count == 0:
                continue
                
            original_values = df[col].copy()
            
            if treatment_method == "drop":
                df_treated = df_treated.dropna(subset=[col])
            else:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if treatment_method == "mean":
                        value = df[col].mean()
                    elif treatment_method == "median":
                        value = df[col].median()
                    else:  # mode
                        value = df[col].mode()[0]
                else:  # categorical
                    value = df[col].mode()[0]
                
                df_treated[col].fillna(value, inplace=True)
            
            treatment_stats["treated_columns"][col] = {
                "missing_count": int(missing_count),
                "treatment_method": treatment_method,
                "replacement_value": value if treatment_method != "drop" else None
            }
        
        # Save treated dataframe and update summary
        new_file_id = db.save_uploaded_file(df_treated, f"treated_{file_id}.csv")
        
        treatment_stats.update({
            "original_file_id": file_id,
            "treated_file_id": new_file_id,
            "original_rows": len(df),
            "treated_rows": len(df_treated),
            "treatment_method": treatment_method,
            "treatment_date": datetime.now().isoformat()
        })
        
        db.add_cleaning_step(file_id, "missing_values", treatment_stats)
        
        return JSONResponse(content=treatment_stats)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

from utils.ai_utils import generate_insight
from utils.serialization import infer_and_convert_types, to_json_serializable
from backend.routes.analytics import router as analytics_router

# Include analytics routes
app.include_router(analytics_router, prefix="/analytics", tags=["Analytics"])

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
        
        # Infer and convert data types
        df = infer_and_convert_types(df)
        
        # Generate summary statistics
        summary = {
            "info": {
                "shape": df.shape,
                "columns": df.columns,
                "dtypes": df.dtypes,
                "missing_values": df.isnull().sum(),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "data_preview": df.head()
            },
            "describe": df.describe(include='all'),
            "missing_analysis": {
                "total_missing": df.isnull().sum().sum(),
                "missing_by_column": df.isnull().sum(),
                "missing_percentage": (df.isnull().sum() / len(df) * 100)
            },
            "numeric_columns": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        }
        
        # Generate AI insights about the dataset
        insight_prompt = f"""
        Given the following dataset summary, provide a brief overview of the data:
        - {df.shape[0]} rows and {df.shape[1]} columns
        - Column types: {df.dtypes.value_counts().to_dict()}
        - Missing values: {df.isnull().sum().sum()} total
        
        Focus on:
        1. Data quality issues (missing values, potential data type conversions needed)
        2. Recommendations for cleaning and preprocessing
        3. Initial observations about the data structure
        """
        
        try:
            summary["ai_insights"] = generate_insight(insight_prompt, context=summary)
        except Exception as e:
            summary["ai_insights"] = f"AI insights generation failed: {str(e)}"
        
        # Generate unique ID and store in database
        file_id = db.save_uploaded_file(df, file.filename)
        db.save_file_summary(file_id, summary)
        
        # Update file status
        db.update_file_status(file_id, "ready")
        
        response_data = {
            "file_id": file_id, 
            "filename": file.filename, 
            "status": "ready",
            "summary": summary
        }
        
        return JSONResponse(content=to_json_serializable(response_data))
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
