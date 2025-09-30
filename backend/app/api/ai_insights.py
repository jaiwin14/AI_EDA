"""
AI Insights API Endpoints
Centralized endpoints for AI-powered analysis across all EDA steps
"""
from fastapi import APIRouter, HTTPException, Depends, Query, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum

# Assuming these imports are correctly set up in your project
from app.core.config import settings
from app.core.database import db_manager, local_storage
from app.api.upload import load_dataset
from ..ml.ai_agent import ai_agent, AnalysisType
# from fastapi.responses import JSONResponse # Already imported
from ..ml.eda_processor import EDAProcessor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai-insights", tags=["AI Insights"])

# Initialize EDA Processor
eda_processor = EDAProcessor()

async def get_dataset(dataset_id: str) -> dict:
    """Load dataset by ID from database or local storage"""
    try:
        # Try to load metadata from local storage first
        try:
            dataset_metadata = local_storage.load_json(f"dataset_{dataset_id}")
            logger.info(f"Loaded dataset metadata for {dataset_id}")
            
            # Check if we have the actual CSV file
            csv_filename = dataset_metadata.get('filename', f"{dataset_id}.csv")
            
            # Try to load the CSV data from temp directory or local storage
            csv_paths = [
                f"temp/{csv_filename}",
                f"temp/{dataset_metadata.get('original_filename', '')}",
                f"local_storage/{csv_filename}",
                f"{dataset_id}.csv"
            ]
            
            df = None
            for csv_path in csv_paths:
                try:
                    full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), csv_path)
                    if os.path.exists(full_path):
                        df = pd.read_csv(full_path)
                        logger.info(f"Loaded CSV data from {full_path}, shape: {df.shape}")
                        break
                except Exception as e:
                    logger.debug(f"Failed to load CSV from {csv_path}: {e}")
                    continue
            
            if df is not None and not df.empty:
                return {"data": df.to_dict(orient='records'), "columns": list(df.columns)}
            else:
                logger.warning(f"No CSV data found for dataset {dataset_id}")
                logger.info(f"Searched paths: {csv_paths}")
                
                # Generate synthetic data based on metadata for demonstration
                columns = dataset_metadata.get('metadata', {}).get('columns', [])
                dtypes = dataset_metadata.get('metadata', {}).get('dtypes', {})
                shape = dataset_metadata.get('metadata', {}).get('shape', [0, 0])
                
                if len(columns) > 0 and shape[0] > 0:
                    logger.info(f"Generating synthetic data with {shape[0]} rows and {len(columns)} columns")
                    
                    # Create synthetic data based on column types
                    synthetic_data = []
                    for i in range(min(100, shape[0])):  # Limit to 100 rows for demo
                        row = {}
                        for col in columns:
                            dtype = dtypes.get(col, 'object')
                            if 'float' in dtype:
                                row[col] = np.random.uniform(0, 100)
                            elif 'int' in dtype:
                                row[col] = np.random.randint(0, 100)
                            elif 'bool' in dtype:
                                row[col] = np.random.choice([True, False])
                            else:  # object/string
                                if 'Dosha' in col:
                                    row[col] = np.random.choice(['Vata', 'Pitta', 'Kapha'])
                                else:
                                    row[col] = f"Value_{i}"
                        synthetic_data.append(row)
                    
                    return {"data": synthetic_data, "columns": columns}
                else:
                    return {"data": [], "columns": columns}
                
        except FileNotFoundError:
            logger.warning(f"Dataset metadata not found for {dataset_id}")
        
        # Try to load from database as fallback
        dataset = await db_manager.get_dataset(dataset_id)
        if dataset:
            # If data is already in the expected format, return it
            if isinstance(dataset, dict) and 'data' in dataset:
                return dataset
                
            # Convert to the expected format if needed
            if isinstance(dataset, dict):
                df = pd.DataFrame(dataset)
            else:
                df = pd.DataFrame(dataset)
            
            return {"data": df.to_dict(orient='records'), "columns": list(df.columns)}
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading dataset: {str(e)}"
        )

async def _perform_analysis(
    dataset_id: str,
    analysis_type: AnalysisType,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Helper function to perform analysis and return results"""
    context = context or {}
    
    try:
        # Get the dataset and convert to DataFrame
        dataset = await get_dataset(dataset_id)
        df = pd.DataFrame(dataset['data'])
        
        if analysis_type == AnalysisType.DATASET_OVERVIEW:
            result = await ai_agent.analyze_dataset_overview(df, dataset_id)
        elif analysis_type == AnalysisType.STATISTICAL_SUMMARY:
            stats = await eda_processor.generate_basic_statistics(df)
            result = await ai_agent.analyze_statistical_summary(df, stats)
        elif analysis_type == AnalysisType.CORRELATION_INSIGHTS:
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) < 2:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="At least two numeric columns are required for correlation analysis"
                )
            corr_matrix = numeric_df.corr()
            
            # Simple correlation analysis without AI agent issues
            correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if not pd.isna(corr_val):
                        correlations.append({
                            "var1": corr_matrix.columns[i],
                            "var2": corr_matrix.columns[j], 
                            "correlation": round(float(corr_val), 3)
                        })
            
            # Sort by absolute correlation value and take top 10
            correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            top_correlations = correlations[:10]
            
            # Create simple insights
            strong_correlations = [c for c in top_correlations if abs(c["correlation"]) > 0.7]
            moderate_correlations = [c for c in top_correlations if 0.3 <= abs(c["correlation"]) <= 0.7]
            
            insights = []
            if strong_correlations:
                insights.append(f"Found {len(strong_correlations)} strong correlations (>0.7)")
                for c in strong_correlations[:3]:
                    direction = "positive" if c["correlation"] > 0 else "negative"
                    insights.append(f"Strong {direction} correlation between {c['var1']} and {c['var2']} ({c['correlation']})")
            
            if moderate_correlations:
                insights.append(f"Found {len(moderate_correlations)} moderate correlations (0.3-0.7)")
            
            insights.append(f"Total numeric variables analyzed: {len(corr_matrix.columns)}")
            insights.append("Consider investigating strong correlations for potential multicollinearity")
            
            result = {
                "success": True,
                "analysis_type": "correlation_insights",
                "data": {
                    "insights": insights,
                    "correlations": top_correlations,
                    "strong_correlations": strong_correlations,
                    "correlation_matrix": corr_matrix.to_dict()
                }
            }
        elif analysis_type == AnalysisType.MISSING_VALUES_ANALYSIS:
            missing_data = {k: int(v) for k, v in df.isnull().sum().to_dict().items()}
            result = await ai_agent.analyze_missing_values(missing_data)
        elif analysis_type == AnalysisType.OUTLIER_ANALYSIS:
            outliers = await eda_processor.detect_outliers(df)
            result = await ai_agent.analyze_outliers(df, outliers)
        elif analysis_type == AnalysisType.DISTRIBUTION_ANALYSIS:
            distributions = await eda_processor.analyze_distributions(df)
            result = await ai_agent.analyze_distributions(distributions)
        elif analysis_type == AnalysisType.DATA_QUALITY_ASSESSMENT:
            result = await ai_agent.analyze_data_quality(df)
        elif analysis_type == AnalysisType.FEATURE_IMPORTANCE:
            feature_importance = await eda_processor.calculate_feature_importance(df)
            result = await ai_agent.analyze_feature_importance(feature_importance)
        elif analysis_type == AnalysisType.TREND_ANALYSIS:
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                result = await ai_agent.analyze_trends(df, date_cols[0])
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No datetime column found for trend analysis"
                )
        elif analysis_type == AnalysisType.BUSINESS_INSIGHTS:
            result = await ai_agent.generate_business_insights(df, context.get('business_context'))
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported analysis type: {analysis_type}"
            )
        
        # Ensure the result is JSON serializable
        if isinstance(result, (pd.DataFrame, pd.Series)):
            result = result.to_dict(orient='records' if isinstance(result, pd.DataFrame) else 'list')
        
        return {
            "success": True,
            "analysis_type": analysis_type.value if hasattr(analysis_type, 'value') else str(analysis_type),
            "dataset_id": dataset_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing analysis {analysis_type} for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating insights: {str(e)}"
        )

@router.get("/{dataset_id}", response_model=Dict[str, Any])
async def get_ai_insights(
    dataset_id: str,
    analysis_type: AnalysisType = Query(..., description="Type of analysis to perform"),
    custom_context: Optional[str] = Query(None, description="Additional context as JSON string")
):
    """
    Generate AI insights for a specific analysis type (query parameter version)
    
    Args:
        dataset_id: ID of the dataset to analyze
        analysis_type: Type of analysis to perform
        custom_context: Additional context as JSON string
    """
    try:
        context = json.loads(custom_context) if custom_context else {}
        return await _perform_analysis(dataset_id, analysis_type, context)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON in custom_context"
        )

# Path-based endpoints for frontend compatibility
@router.get("/{dataset_id}/overview", response_model=Dict[str, Any])
async def get_dataset_overview(dataset_id: str):
    """Get AI-powered dataset overview insights"""
    return await _perform_analysis(dataset_id, AnalysisType.DATASET_OVERVIEW)

@router.get("/{dataset_id}/statistics", response_model=Dict[str, Any])
async def get_statistical_insights(dataset_id: str):
    """Get AI-powered statistical insights"""
    return await _perform_analysis(dataset_id, AnalysisType.STATISTICAL_SUMMARY)

@router.get("/{dataset_id}/correlations", response_model=Dict[str, Any])
async def get_correlation_insights(dataset_id: str):
    """Get AI-powered correlation insights"""
    return await _perform_analysis(dataset_id, AnalysisType.CORRELATION_INSIGHTS)

@router.get("/{dataset_id}/missing-values", response_model=Dict[str, Any])
async def get_missing_values_insights(dataset_id: str):
    """Get AI-powered missing values insights"""
    return await _perform_analysis(dataset_id, AnalysisType.MISSING_VALUES_ANALYSIS)

@router.get("/{dataset_id}/outliers", response_model=Dict[str, Any])
async def get_outlier_insights(dataset_id: str):
    """Get AI-powered outlier insights"""
    return await _perform_analysis(dataset_id, AnalysisType.OUTLIER_ANALYSIS)

@router.get("/{dataset_id}/distributions", response_model=Dict[str, Any])
async def get_distribution_insights(dataset_id: str):
    """Get AI-powered distribution insights"""
    return await _perform_analysis(dataset_id, AnalysisType.DISTRIBUTION_ANALYSIS)

@router.get("/{dataset_id}/data-quality", response_model=Dict[str, Any])
async def get_data_quality_insights(dataset_id: str):
    """Get AI-powered data quality assessment"""
    return await _perform_analysis(dataset_id, AnalysisType.DATA_QUALITY_ASSESSMENT)

@router.get("/{dataset_id}/features", response_model=Dict[str, Any])
async def get_feature_importance_insights(dataset_id: str):
    """Get AI-powered feature importance insights"""
    return await _perform_analysis(dataset_id, AnalysisType.FEATURE_IMPORTANCE)

@router.get("/{dataset_id}/trends", response_model=Dict[str, Any])
async def get_trend_insights(dataset_id: str):
    """Get AI-powered trend analysis insights"""
    return await _perform_analysis(dataset_id, AnalysisType.TREND_ANALYSIS)

@router.get("/{dataset_id}/business", response_model=Dict[str, Any])
async def get_business_insights(
    dataset_id: str,
    context: Optional[str] = Query(None, description="Additional context as JSON string")
):
    """Get AI-powered business insights"""
    custom_context = json.loads(context) if context else {}
    return await _perform_analysis(
        dataset_id, 
        AnalysisType.BUSINESS_INSIGHTS,
        {"business_context": custom_context}
    )
        
@router.get("/{dataset_id}/templates")
async def get_available_templates():
    """Get list of available AI analysis templates"""
    return {
        "templates": [
            {
                "id": AnalysisType.DATASET_OVERVIEW.value,
                "name": "Dataset Overview",
                "description": "Get a comprehensive overview of the dataset"
            },
            {
                "id": AnalysisType.STATISTICAL_SUMMARY.value,
                "name": "Statistical Summary",
                "description": "Detailed statistical analysis of the dataset"
            },
            {
                "id": AnalysisType.CORRELATION_INSIGHTS.value,
                "name": "Correlation Analysis",
                "description": "Analyze relationships between numeric features"
            },
            {
                "id": AnalysisType.MISSING_VALUES_ANALYSIS.value,
                "name": "Missing Values Analysis",
                "description": "Identify and analyze missing data patterns"
            },
            {
                "id": AnalysisType.OUTLIER_ANALYSIS.value,
                "name": "Outlier Detection",
                "description": "Identify and analyze outliers in the data"
            },
            {
                "id": AnalysisType.DISTRIBUTION_ANALYSIS.value,
                "name": "Distribution Analysis",
                "description": "Analyze feature distributions"
            },
            {
                "id": AnalysisType.DATA_QUALITY_ASSESSMENT.value,
                "name": "Data Quality Assessment",
                "description": "Evaluate overall data quality"
            },
            {
                "id": AnalysisType.FEATURE_IMPORTANCE.value,
                "name": "Feature Importance",
                "description": "Analyze feature importance for modeling"
            },
            {
                "id": AnalysisType.TREND_ANALYSIS.value,
                "name": "Trend Analysis",
                "description": "Analyze trends over time"
            },
            {
                "id": AnalysisType.BUSINESS_INSIGHTS.value,
                "name": "Business Insights",
                "description": "Generate business-focused insights"
            }
        ]
    }

@router.post("/{dataset_id}/custom")
async def generate_custom_insights(
    dataset_id: str,
    template_type: str,
    context_data: Dict[str, Any]
):
    """
    Generate custom AI insights with specific template and context
    
    Args:
        dataset_id: Dataset identifier
        template_type: Prompt template type to use (from /templates endpoint)
        context_data: Additional context data for the analysis
    """
    try:
        # Get the dataset
        dataset = await get_dataset(dataset_id) # Call get_dataset to retrieve data
        df = pd.DataFrame(dataset['data'])
        
        # Add dataset info to context
        context = {
            "dataset_info": {
                "rows": len(df),
                "columns": list(df.columns),
                "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
            },
            **context_data
        }
        
        # Generate insights using the AI agent
        # We need to map the string template_type back to an AnalysisType enum if ai_agent.analyze expects it,
        # or ensure ai_agent.analyze can take a string directly.
        # Assuming ai_agent.analyze expects an AnalysisType.value string.
        result = await ai_agent.analyze(template_type, context)
        
        return {
            "success": True,
            "template_type": template_type,
            "dataset_id": dataset_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating custom insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate custom insights: {str(e)}")