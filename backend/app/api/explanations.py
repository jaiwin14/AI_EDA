"""
Model Explanation API Endpoints
SHAP-based model explanations and interpretability
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel

from app.core.config import settings
from app.core.database import db_manager, local_storage
from app.ml.explainer import ModelExplainer
from app.api.upload import load_dataset

logger = logging.getLogger(__name__)
router = APIRouter()

class ExplanationRequest(BaseModel):
    model_id: str
    input_data: Dict[str, Union[str, int, float]]
    explanation_type: Optional[str] = "local"  # local, global, summary

class BatchExplanationRequest(BaseModel):
    model_id: str
    input_data: List[Dict[str, Union[str, int, float]]]
    explanation_type: Optional[str] = "local"

@router.post("/explain")
async def explain_prediction(request: ExplanationRequest) -> Dict[str, Any]:
    """
    Generate SHAP explanations for a single prediction
    
    Args:
        request: Explanation request with model ID and input data
    """
    
    try:
        # Get model metadata
        model_metadata = await db_manager.get_model_metadata(request.model_id)
        
        if not model_metadata:
            # Fallback to local storage search
            model_files = list(settings.MODELS_DIR.glob(f"*{request.model_id}*"))
            if not model_files:
                raise HTTPException(
                    status_code=404,
                    detail="Model not found"
                )
            model_path = str(model_files[0])
        else:
            model_path = model_metadata["model_path"]
        
        # Initialize explainer
        explainer = ModelExplainer()
        
        # Prepare input data
        input_df = pd.DataFrame([request.input_data])
        
        # Initialize explainer (without background data for now)
        init_result = await explainer.initialize_explainer(model_path)
        
        if init_result["status"] != "success":
            raise HTTPException(
                status_code=500,
                detail=init_result["message"]
            )
        
        # Generate explanation
        explanation = await explainer.explain_prediction(
            input_df, 
            request.explanation_type
        )
        
        if explanation["status"] != "success":
            raise HTTPException(
                status_code=500,
                detail=explanation["message"]
            )
        
        return {
            "model_id": request.model_id,
            "explanation_type": request.explanation_type,
            "input_data": request.input_data,
            "explanation": explanation,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Explanation failed for model {request.model_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Explanation failed: {str(e)}"
        )

@router.post("/explain/batch")
async def explain_batch_predictions(request: BatchExplanationRequest) -> Dict[str, Any]:
    """
    Generate SHAP explanations for batch predictions
    
    Args:
        request: Batch explanation request
    """
    
    try:
        # Get model metadata
        model_metadata = await db_manager.get_model_metadata(request.model_id)
        
        if not model_metadata:
            # Fallback to local storage search
            model_files = list(settings.MODELS_DIR.glob(f"*{request.model_id}*"))
            if not model_files:
                raise HTTPException(
                    status_code=404,
                    detail="Model not found"
                )
            model_path = str(model_files[0])
        else:
            model_path = model_metadata["model_path"]
        
        # Initialize explainer
        explainer = ModelExplainer()
        
        # Prepare input data
        input_df = pd.DataFrame(request.input_data)
        
        # Initialize explainer
        init_result = await explainer.initialize_explainer(model_path)
        
        if init_result["status"] != "success":
            raise HTTPException(
                status_code=500,
                detail=init_result["message"]
            )
        
        # Generate explanations
        explanations = await explainer.explain_prediction(
            input_df, 
            request.explanation_type
        )
        
        if explanations["status"] != "success":
            raise HTTPException(
                status_code=500,
                detail=explanations["message"]
            )
        
        return {
            "model_id": request.model_id,
            "explanation_type": request.explanation_type,
            "total_explanations": len(request.input_data),
            "explanations": explanations,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Batch explanation failed for model {request.model_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch explanation failed: {str(e)}"
        )

@router.post("/explain/{model_id}/plots")
async def generate_explanation_plots(
    model_id: str,
    input_data: List[Dict[str, Union[str, int, float]]]
) -> Dict[str, Any]:
    """
    Generate SHAP visualization plots
    
    Args:
        model_id: ID of the trained model
        input_data: Input data for generating plots
    """
    
    try:
        # Get model metadata
        model_metadata = await db_manager.get_model_metadata(model_id)
        
        if not model_metadata:
            # Fallback to local storage search
            model_files = list(settings.MODELS_DIR.glob(f"*{model_id}*"))
            if not model_files:
                raise HTTPException(
                    status_code=404,
                    detail="Model not found"
                )
            model_path = str(model_files[0])
        else:
            model_path = model_metadata["model_path"]
        
        # Initialize explainer
        explainer = ModelExplainer()
        
        # Prepare input data
        input_df = pd.DataFrame(input_data)
        
        # Initialize explainer
        init_result = await explainer.initialize_explainer(model_path)
        
        if init_result["status"] != "success":
            raise HTTPException(
                status_code=500,
                detail=init_result["message"]
            )
        
        # Generate plots
        plots = await explainer.generate_explanation_plots(input_df)
        
        if plots["status"] != "success":
            raise HTTPException(
                status_code=500,
                detail=plots["message"]
            )
        
        return {
            "model_id": model_id,
            "plots": plots,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Plot generation failed for model {model_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Plot generation failed: {str(e)}"
        )

@router.get("/explain/{model_id}/global")
async def get_global_explanations(
    model_id: str,
    dataset_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get global model explanations using training data or specified dataset
    
    Args:
        model_id: ID of the trained model
        dataset_id: Optional dataset ID to use for global explanations
    """
    
    try:
        # Get model metadata
        model_metadata = await db_manager.get_model_metadata(model_id)
        
        if not model_metadata:
            raise HTTPException(
                status_code=404,
                detail="Model not found"
            )
        
        model_path = model_metadata["model_path"]
        
        # Get dataset for global explanations
        if dataset_id:
            # Use specified dataset
            dataset_files = list(settings.UPLOAD_DIR.glob(f"{dataset_id}.*"))
            if not dataset_files:
                raise HTTPException(
                    status_code=404,
                    detail="Dataset not found"
                )
            df = await load_dataset(dataset_files[0])
        else:
            # Use model's training dataset
            training_dataset_id = model_metadata.get("dataset_id")
            if not training_dataset_id:
                raise HTTPException(
                    status_code=400,
                    detail="No dataset specified and model has no associated training dataset"
                )
            
            dataset_files = list(settings.UPLOAD_DIR.glob(f"{training_dataset_id}.*"))
            if not dataset_files:
                raise HTTPException(
                    status_code=404,
                    detail="Training dataset not found"
                )
            df = await load_dataset(dataset_files[0])
        
        # Remove target column if present
        target_column = model_metadata.get("target_column")
        if target_column and target_column in df.columns:
            df = df.drop(columns=[target_column])
        
        # Sample data for efficiency (use first 1000 rows)
        sample_df = df.head(1000)
        
        # Initialize explainer with background data
        explainer = ModelExplainer()
        init_result = await explainer.initialize_explainer(model_path, sample_df)
        
        if init_result["status"] != "success":
            raise HTTPException(
                status_code=500,
                detail=init_result["message"]
            )
        
        # Generate global explanations
        global_explanation = await explainer.explain_prediction(
            sample_df, 
            "global"
        )
        
        if global_explanation["status"] != "success":
            raise HTTPException(
                status_code=500,
                detail=global_explanation["message"]
            )
        
        return {
            "model_id": model_id,
            "dataset_used": dataset_id or training_dataset_id,
            "sample_size": len(sample_df),
            "global_explanation": global_explanation,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Global explanation failed for model {model_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Global explanation failed: {str(e)}"
        )
