"""
File Upload API Endpoints
Handle CSV, Excel, and Parquet file uploads with validation
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from pathlib import Path
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List
import os

from app.core.config import settings
from app.core.database import db_manager, local_storage
from app.ml.data_validator import DataValidator
from app.utils.file_utils import save_upload_file, get_file_info
from app.utils.json_utils import serialize_for_json

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/upload")
async def upload_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Upload and validate a dataset file
    
    Supports: CSV, Excel (.xlsx, .xls), Parquet
    Returns: Dataset ID and basic metadata
    """
    
    # Validate file type
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in settings.ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types: {settings.ALLOWED_FILE_TYPES}"
        )
    
    # Validate file size
    file_size = 0
    content = await file.read()
    file_size = len(content)
    
    if file_size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB"
        )
    
    # Generate unique filename
    dataset_id = str(uuid.uuid4())
    filename = f"{dataset_id}{file_extension}"
    filepath = settings.UPLOAD_DIR / filename
    
    try:
        # Save file
        await save_upload_file(content, filepath)
        
        # Load and validate dataset
        df = await load_dataset(filepath)
        
        # Basic validation
        validator = DataValidator()
        validation_results = await validator.validate_dataset(df)
        
        if not validation_results["is_valid"]:
            # Clean up file if validation fails
            if filepath.exists():
                os.remove(filepath)
            raise HTTPException(
                status_code=400,
                detail=f"Dataset validation failed: {validation_results['errors']}"
            )
        
        # Get dataset metadata with JSON serialization fixes
        metadata = serialize_for_json({
            "shape": [df.shape[0], df.shape[1]],
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "validation_results": validation_results,
            "upload_timestamp": datetime.utcnow().isoformat()
        })
        
        # Save metadata to database
        db_dataset_id = await db_manager.save_dataset_metadata(
            filename=filename,
            original_filename=file.filename,
            file_size=file_size,
            rows=df.shape[0],
            cols=df.shape[1],
            metadata=metadata
        )
        
        # Fallback to local storage if database save fails
        if not db_dataset_id:
            local_storage.save_json(f"dataset_{dataset_id}", {
                "dataset_id": dataset_id,
                "filename": filename,
                "original_filename": file.filename,
                "metadata": metadata
            })
        
        # Schedule background EDA processing
        background_tasks.add_task(
            process_dataset_background,
            dataset_id,
            filepath
        )
        
        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "file_size": file_size,
            "size": file_size,  # Keep for backward compatibility
            "rows": df.shape[0],
            "columns": df.shape[1],
            "upload_time": datetime.utcnow().isoformat(),
            "status": "uploaded",
            "message": "Dataset uploaded successfully. EDA processing started in background.",
            "validation": validation_results,
            "metadata": {
                "shape": [df.shape[0], df.shape[1]],
                "columns": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "missing_values": df.isnull().sum().to_dict()
            }
        }
        
    except Exception as e:
        # Clean up file on error
        if filepath.exists():
            os.remove(filepath)
        
        logger.error(f"Upload failed for {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Upload processing failed: {str(e)}"
        )

@router.get("/upload/datasets")
async def list_datasets() -> List[Dict[str, Any]]:
    """List all uploaded datasets"""
    
    # Try to get from database first
    try:
        datasets = await db_manager.list_datasets()
        if datasets:
            return datasets
    except Exception as e:
        logger.warning(f"Database query failed: {str(e)}")
    
    # Fallback to scanning upload directory
    try:
        datasets = []
        for filepath in settings.UPLOAD_DIR.glob("*"):
            if filepath.is_file() and filepath.suffix.lower() in settings.ALLOWED_FILE_TYPES:
                dataset_id = filepath.stem
                
                # Try to load metadata
                metadata = local_storage.load_json(f"dataset_{dataset_id}")
                if metadata:
                    # Get file stats
                    file_stats = filepath.stat()
                    upload_time = metadata.get("metadata", {}).get("upload_timestamp")
                    if not upload_time:
                        upload_time = datetime.fromtimestamp(file_stats.st_ctime).isoformat()
                    
                    datasets.append({
                        "dataset_id": dataset_id,
                        "filename": metadata.get("original_filename", filepath.name),
                        "file_size": file_stats.st_size,
                        "upload_time": upload_time,
                        "status": "completed",
                        "rows": metadata.get("metadata", {}).get("shape", [0, 0])[0],
                        "columns": metadata.get("metadata", {}).get("shape", [0, 0])[1]
                    })
        
        # Sort by upload time (newest first)
        datasets.sort(key=lambda x: x.get("upload_time", ""), reverse=True)
        return datasets
        
    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to list datasets"
        )

@router.get("/upload/datasets/{dataset_id}")
async def get_dataset_info(dataset_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific dataset"""
    
    # Try database first
    metadata = await db_manager.get_dataset_metadata(dataset_id)
    
    if not metadata:
        # Fallback to local storage
        metadata = local_storage.load_json(f"dataset_{dataset_id}")
    
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found"
        )
    
    return metadata

@router.delete("/upload/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str) -> Dict[str, str]:
    """Delete a dataset and all associated files"""
    
    try:
        # Find and delete file
        for filepath in settings.UPLOAD_DIR.glob(f"{dataset_id}.*"):
            if filepath.is_file():
                os.remove(filepath)
        
        # Delete from local storage
        metadata_file = settings.BASE_DIR / "local_storage" / f"dataset_{dataset_id}.json"
        if metadata_file.exists():
            os.remove(metadata_file)
        
        # TODO: Delete from database when implemented
        
        return {"message": "Dataset deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete dataset"
        )

async def load_dataset(filepath: Path) -> pd.DataFrame:
    """Load dataset from file with appropriate reader"""
    
    try:
        file_extension = filepath.suffix.lower()
        
        if file_extension == '.csv':
            # Try different encodings and separators
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(filepath, encoding=encoding, nrows=settings.MAX_ROWS_FOR_PROCESSING)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not decode CSV file with any supported encoding")
                
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath, nrows=settings.MAX_ROWS_FOR_PROCESSING)
            
        elif file_extension == '.parquet':
            df = pd.read_parquet(filepath)
            if len(df) > settings.MAX_ROWS_FOR_PROCESSING:
                df = df.head(settings.MAX_ROWS_FOR_PROCESSING)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset from {filepath}: {str(e)}")
        raise ValueError(f"Failed to load dataset: {str(e)}")

async def process_dataset_background(dataset_id: str, filepath: Path):
    """Background task to process dataset after upload"""
    
    try:
        logger.info(f"Starting background processing for dataset {dataset_id}")
        
        # Load dataset
        df = await load_dataset(filepath)
        
        # Run basic EDA
        from app.ml.eda_pipeline import EDAProcessor
        eda_processor = EDAProcessor()
        
        # Basic profiling
        basic_stats = await eda_processor.generate_basic_statistics(df)
        await db_manager.save_eda_results(dataset_id, "basic_stats", basic_stats)
        
        # Missing values analysis
        missing_analysis = await eda_processor.analyze_missing_values(df)
        await db_manager.save_eda_results(dataset_id, "missing_values", missing_analysis)
        
        # Correlation analysis
        correlation_analysis = await eda_processor.analyze_correlations(df)
        await db_manager.save_eda_results(dataset_id, "correlations", correlation_analysis)
        
        logger.info(f"Background processing completed for dataset {dataset_id}")
        
    except Exception as e:
        logger.error(f"Background processing failed for dataset {dataset_id}: {str(e)}")
