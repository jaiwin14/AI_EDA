"""
Enhanced Correlation Insights API Endpoint
Provides correlation analysis with Gemini AI insights
"""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json

from app.core.config import settings
from app.core.database import db_manager, local_storage
from app.api.upload import load_dataset
from ..ml.eda_processor import EDAProcessor
from ..ml.visualization_generator import VisualizationGenerator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/correlation", tags=["Correlation Analysis"])

@router.get("/{dataset_id}/analysis")
async def get_enhanced_correlation_analysis(dataset_id: str) -> Dict[str, Any]:
    """
    Get comprehensive correlation analysis with AI insights
    
    Args:
        dataset_id: ID of the dataset to analyze
        
    Returns:
        Dict containing correlation matrix, insights, and visualizations
    """
    try:
        # Load dataset from local storage or database
        dataset_data = None
        
        # Try local storage first
        try:
            dataset_metadata = local_storage.load_json(f"dataset_{dataset_id}")
            logger.info(f"Loaded dataset metadata for {dataset_id}")
            
            # Try to load the actual CSV file
            csv_filename = dataset_metadata.get('filename', f"{dataset_id}.csv")
            csv_paths = [
                f"temp/{csv_filename}",
                f"temp/{dataset_metadata.get('original_filename', '')}",
                f"local_storage/{csv_filename}",
                f"{dataset_id}.csv"
            ]
            
            df = None
            for csv_path in csv_paths:
                try:
                    import os
                    full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), csv_path)
                    if os.path.exists(full_path):
                        df = pd.read_csv(full_path)
                        logger.info(f"Loaded CSV data from {full_path}, shape: {df.shape}")
                        break
                except Exception as e:
                    logger.debug(f"Failed to load CSV from {csv_path}: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Could not load from local storage: {e}")
            
        # Fallback to database
        if df is None:
            try:
                dataset_data = await db_manager.get_dataset(dataset_id)
                if dataset_data:
                    if isinstance(dataset_data, dict) and 'data' in dataset_data:
                        df = pd.DataFrame(dataset_data['data'])
                    else:
                        df = pd.DataFrame(dataset_data)
                    logger.info(f"Loaded dataset from database, shape: {df.shape}")
            except Exception as e:
                logger.error(f"Failed to load from database: {e}")
        
        if df is None or df.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found or empty"
            )
        
        # Initialize processors
        eda_processor = EDAProcessor()
        viz_generator = VisualizationGenerator()
        
        # Perform correlation analysis
        correlation_analysis = await eda_processor.analyze_correlations(df)
        
        # Generate correlation heatmap if we have valid correlations
        correlation_viz = None
        if correlation_analysis.get("numeric_columns") and len(correlation_analysis["numeric_columns"]) >= 2:
            try:
                correlation_viz = await viz_generator.create_correlation_heatmap(df)
            except Exception as viz_error:
                logger.warning(f"Failed to generate correlation visualization: {viz_error}")
                correlation_viz = None
        
        # Prepare response
        response = {
            "success": True,
            "dataset_id": dataset_id,
            "timestamp": datetime.utcnow().isoformat(),
            "correlation_analysis": correlation_analysis,
            "visualization": correlation_viz,
            "dataset_info": {
                "shape": list(df.shape),
                "columns": list(df.columns),
                "numeric_columns": correlation_analysis.get("numeric_columns", []),
                "total_columns": len(df.columns)
            }
        }
        
        # Add AI insights if we have the AI agent available
        try:
            from ..ml.ai_agent import ai_agent
            
            if correlation_analysis.get("numeric_columns") and len(correlation_analysis["numeric_columns"]) >= 2:
                # Prepare context for AI analysis
                corr_matrix = pd.DataFrame.from_dict(correlation_analysis["correlation_matrix"])
                ai_insights = await ai_agent.analyze_correlations(corr_matrix)
                response["ai_insights"] = ai_insights
            else:
                response["ai_insights"] = {
                    "success": False,
                    "message": "Insufficient numeric data for AI correlation insights"
                }
                
        except Exception as ai_error:
            logger.warning(f"Failed to generate AI insights: {ai_error}")
            response["ai_insights"] = {
                "success": False,
                "error": str(ai_error)
            }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in correlation analysis for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform correlation analysis: {str(e)}"
        )

@router.get("/{dataset_id}/matrix")
async def get_correlation_matrix(dataset_id: str) -> Dict[str, Any]:
    """
    Get just the correlation matrix data
    
    Args:
        dataset_id: ID of the dataset
        
    Returns:
        Dict containing correlation matrix
    """
    try:
        # This is a simplified version that just returns the correlation matrix
        analysis = await get_enhanced_correlation_analysis(dataset_id)
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "correlation_matrix": analysis["correlation_analysis"].get("correlation_matrix", {}),
            "numeric_columns": analysis["correlation_analysis"].get("numeric_columns", []),
            "strong_correlations": analysis["correlation_analysis"].get("strong_correlations", [])
        }
        
    except Exception as e:
        logger.error(f"Error getting correlation matrix for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get correlation matrix: {str(e)}"
        )

@router.get("/{dataset_id}/insights")
async def get_correlation_insights_only(dataset_id: str) -> Dict[str, Any]:
    """
    Get just the AI insights for correlations
    
    Args:
        dataset_id: ID of the dataset
        
    Returns:
        Dict containing AI insights
    """
    try:
        analysis = await get_enhanced_correlation_analysis(dataset_id)
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "ai_insights": analysis.get("ai_insights", {}),
            "correlation_summary": analysis["correlation_analysis"].get("correlation_summary", {})
        }
        
    except Exception as e:
        logger.error(f"Error getting correlation insights for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get correlation insights: {str(e)}"
        )
