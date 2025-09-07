"""
Prediction API Endpoints
Make predictions using trained models
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel
import joblib

from app.core.config import settings
from app.core.database import db_manager, local_storage
from app.ml.model_trainer import ModelTrainer

logger = logging.getLogger(__name__)
router = APIRouter()

class PredictionRequest(BaseModel):
    model_id: str
    input_data: Dict[str, Union[str, int, float]]

class BatchPredictionRequest(BaseModel):
    model_id: str
    input_data: List[Dict[str, Union[str, int, float]]]

@router.post("/predict")
async def make_prediction(request: PredictionRequest) -> Dict[str, Any]:
    """
    Make a single prediction using a trained model
    
    Args:
        request: Prediction request with model ID and input data
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
        
        # Load model
        model = joblib.load(model_path)
        
        # Prepare input data
        input_df = pd.DataFrame([request.input_data])
        
        # Make prediction
        prediction = model.predict(input_df)
        
        # Get prediction probabilities for classification
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(input_df)[0].tolist()
            except:
                pass
        
        # Prepare response
        result = {
            "model_id": request.model_id,
            "prediction": prediction[0].tolist() if isinstance(prediction[0], np.ndarray) else prediction[0],
            "probabilities": probabilities,
            "input_data": request.input_data,
            "status": "success"
        }
        
        # Save prediction to database
        await db_manager.save_prediction(
            model_id=request.model_id,
            input_data=request.input_data,
            prediction={"value": result["prediction"], "probabilities": probabilities},
            confidence=max(probabilities) if probabilities else None
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed for model {request.model_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@router.post("/batch_predict")
async def make_batch_predictions(request: BatchPredictionRequest) -> Dict[str, Any]:
    """
    Make batch predictions using a trained model
    
    Args:
        request: Batch prediction request with model ID and input data list
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
        
        # Load model
        model = joblib.load(model_path)
        
        # Prepare input data
        input_df = pd.DataFrame(request.input_data)
        
        # Make predictions
        predictions = model.predict(input_df)
        
        # Get prediction probabilities for classification
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(input_df).tolist()
            except:
                pass
        
        # Prepare response
        results = []
        for i, (pred, input_row) in enumerate(zip(predictions, request.input_data)):
            result = {
                "prediction": pred.tolist() if isinstance(pred, np.ndarray) else pred,
                "probabilities": probabilities[i] if probabilities else None,
                "input_data": input_row
            }
            results.append(result)
        
        return {
            "model_id": request.model_id,
            "predictions": results,
            "total_predictions": len(results),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Batch prediction failed for model {request.model_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

@router.get("/predict/{model_id}/schema")
async def get_prediction_schema(model_id: str) -> Dict[str, Any]:
    """
    Get the input schema for making predictions with a model
    
    Args:
        model_id: ID of the trained model
    """
    
    try:
        # Get model metadata
        model_metadata = await db_manager.get_model_metadata(model_id)
        
        if not model_metadata:
            raise HTTPException(
                status_code=404,
                detail="Model not found"
            )
        
        features = model_metadata.get("features", [])
        
        # Try to infer feature types from training data
        # This is a simplified schema - in production, you'd store more detailed schema info
        schema = {
            "model_id": model_id,
            "required_features": features,
            "feature_types": {},  # Would be populated from training metadata
            "example_input": {},
            "task_type": model_metadata.get("task_type"),
            "target_column": model_metadata.get("target_column")
        }
        
        # Create example input (simplified)
        for feature in features:
            schema["example_input"][feature] = "value"  # Placeholder
        
        return schema
        
    except Exception as e:
        logger.error(f"Error getting prediction schema for model {model_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get prediction schema"
        )

@router.get("/predict/{model_id}/history")
async def get_prediction_history(
    model_id: str,
    limit: Optional[int] = 100
) -> Dict[str, Any]:
    """
    Get prediction history for a model
    
    Args:
        model_id: ID of the trained model
        limit: Maximum number of predictions to return
    """
    
    try:
        # This would typically query the database
        # For now, return a placeholder response
        return {
            "model_id": model_id,
            "predictions": [],
            "total_predictions": 0,
            "message": "Prediction history feature coming soon"
        }
        
    except Exception as e:
        logger.error(f"Error getting prediction history for model {model_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get prediction history"
        )
