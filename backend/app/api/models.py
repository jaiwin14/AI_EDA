"""
Model Training and Management API Endpoints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from app.core.config import settings
from app.core.database import db_manager, local_storage
from app.ml.model_trainer import ModelTrainer
from app.api.upload import load_dataset

logger = logging.getLogger(__name__)
router = APIRouter()

class TrainingRequest(BaseModel):
    dataset_id: str
    target_column: Optional[str] = None
    models_to_train: Optional[List[str]] = None
    test_size: Optional[float] = 0.2

@router.post("/models/train")
async def train_models(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Train ML models on a dataset
    
    Args:
        request: Training configuration
    """
    
    # Find dataset file
    dataset_files = list(settings.UPLOAD_DIR.glob(f"{request.dataset_id}.*"))
    if not dataset_files:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found"
        )
    
    filepath = dataset_files[0]
    
    try:
        # Load dataset
        df = await load_dataset(filepath)
        
        # Initialize model trainer
        trainer = ModelTrainer()
        
        # Start training in background
        background_tasks.add_task(
            train_models_background,
            request.dataset_id,
            df,
            request.target_column,
            request.models_to_train
        )
        
        return {
            "status": "training_started",
            "dataset_id": request.dataset_id,
            "message": "Model training started in background. Check status for updates.",
            "target_column": request.target_column
        }
        
    except Exception as e:
        logger.error(f"Model training initiation failed for dataset {request.dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start model training: {str(e)}"
        )

@router.get("/models/{dataset_id}")
async def get_trained_models(dataset_id: str) -> Dict[str, Any]:
    """Get all trained models for a dataset"""
    
    try:
        # Get models from database
        models = await db_manager.get_models_for_dataset(dataset_id)
        
        if not models:
            # Fallback to local storage
            models_data = local_storage.load_json(f"models_{dataset_id}")
            if models_data:
                models = models_data.get("models", [])
        
        if not models:
            return {
                "dataset_id": dataset_id,
                "models": [],
                "message": "No trained models found for this dataset"
            }
        
        return {
            "dataset_id": dataset_id,
            "models": models,
            "total_models": len(models)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving models for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve trained models"
        )

@router.get("/models/{dataset_id}/status")
async def get_training_status(dataset_id: str) -> Dict[str, Any]:
    """Get training status for a dataset"""
    
    try:
        # Check local storage for training status
        status_data = local_storage.load_json(f"training_status_{dataset_id}")
        
        if not status_data:
            return {
                "dataset_id": dataset_id,
                "status": "not_started",
                "message": "No training initiated for this dataset"
            }
        
        return status_data
        
    except Exception as e:
        logger.error(f"Error retrieving training status for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve training status"
        )

@router.get("/models/{dataset_id}/comparison")
async def get_model_comparison(dataset_id: str) -> Dict[str, Any]:
    """Get model comparison results"""
    
    try:
        # Get training results
        results_data = local_storage.load_json(f"training_results_{dataset_id}")
        
        if not results_data:
            raise HTTPException(
                status_code=404,
                detail="No training results found for this dataset"
            )
        
        comparison = results_data.get("model_comparison", {})
        
        return {
            "dataset_id": dataset_id,
            "comparison": comparison,
            "best_model": results_data.get("best_model"),
            "task_type": results_data.get("task_type")
        }
        
    except Exception as e:
        logger.error(f"Error retrieving model comparison for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve model comparison"
        )

@router.delete("/models/{model_id}")
async def delete_model(model_id: str) -> Dict[str, str]:
    """Delete a trained model"""
    
    try:
        # Find and delete model file
        model_files = list(settings.MODELS_DIR.glob(f"*{model_id}*"))
        
        for model_file in model_files:
            model_file.unlink()
        
        # TODO: Delete from database when implemented
        
        return {"message": "Model deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting model {model_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete model"
        )

async def train_models_background(dataset_id: str, df: pd.DataFrame, 
                                target_column: Optional[str] = None,
                                models_to_train: Optional[List[str]] = None):
    """Background task for model training"""
    
    try:
        logger.info(f"Starting background model training for dataset {dataset_id}")
        
        # Update status
        status_data = {
            "dataset_id": dataset_id,
            "status": "training",
            "started_at": pd.Timestamp.now().isoformat(),
            "progress": "Initializing training..."
        }
        local_storage.save_json(f"training_status_{dataset_id}", status_data)
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Update progress
        status_data["progress"] = "Detecting task type and preparing data..."
        local_storage.save_json(f"training_status_{dataset_id}", status_data)
        
        # Train models
        results = await trainer.auto_train_models(df, target_column)
        
        # Save results
        local_storage.save_json(f"training_results_{dataset_id}", results)
        
        # Save individual model metadata to database
        for model_name, model_results in results["models"].items():
            if model_results.get("status") == "completed":
                await db_manager.save_model_metadata(
                    dataset_id=dataset_id,
                    model_name=model_name,
                    model_type=model_name,
                    task_type=results["task_type"],
                    target_column=results["target_column"],
                    features=results["feature_columns"],
                    hyperparameters={},
                    metrics=model_results["metrics"],
                    model_path=model_results["model_path"]
                )
        
        # Update final status
        status_data.update({
            "status": "completed",
            "completed_at": pd.Timestamp.now().isoformat(),
            "progress": "Training completed successfully",
            "results_summary": {
                "total_models": len(results["models"]),
                "successful_models": len([m for m in results["models"].values() if m.get("status") == "completed"]),
                "best_model": results.get("best_model"),
                "task_type": results["task_type"]
            }
        })
        local_storage.save_json(f"training_status_{dataset_id}", status_data)
        
        logger.info(f"Background model training completed for dataset {dataset_id}")
        
    except Exception as e:
        logger.error(f"Background model training failed for dataset {dataset_id}: {str(e)}")
        
        # Update error status
        status_data = {
            "dataset_id": dataset_id,
            "status": "failed",
            "error": str(e),
            "failed_at": pd.Timestamp.now().isoformat()
        }
        local_storage.save_json(f"training_status_{dataset_id}", status_data)
