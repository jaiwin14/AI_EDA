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
import os
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum

# Import required modules
from app.core.database import DatabaseManager, local_storage
from app.core.config import settings
from app.ml.ai_agent import ai_agent, AnalysisType
from app.ml.eda_processor import EDAProcessor
from app.ml.missing_values_processor import MissingValuesProcessor
from app.ml.outlier_processor import OutlierProcessor
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai-insights", tags=["AI Insights"])
def clean_gemini_response(response_text: str) -> dict:
    """Clean Gemini API response and parse JSON"""
    
    # Remove markdown code blocks
    if '```json' in response_text:
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'\s*```', '', response_text)
    elif '```' in response_text:
        response_text = re.sub(r'```\s*', '', response_text)
    
    # Strip whitespace
    response_text = response_text.strip()
    
    # Parse JSON
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Fallback: extract JSON object
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        raise
# Initialize processors
eda_processor = EDAProcessor()
missing_values_processor = MissingValuesProcessor()
outlier_processor = OutlierProcessor()

# Replace the get_dataset function in backend/app/api/ai_insights.py
# Around line 36

async def get_dataset(dataset_id: str) -> dict:
    """Load dataset by ID from database or local storage - FIXED VERSION"""
    try:
        logger.info(f"Loading dataset {dataset_id}")
        
        # OPTION 1: Try to load from actual uploaded file (MOST RELIABLE)
        from pathlib import Path
        from app.core.config import settings
        from app.api.upload import load_dataset
        
        # Look for the actual uploaded CSV file
        dataset_files = list(settings.UPLOAD_DIR.glob(f"{dataset_id}.*"))
        
        if dataset_files:
            filepath = dataset_files[0]
            logger.info(f"Found dataset file: {filepath}")
            df = await load_dataset(filepath)
            logger.info(f"Successfully loaded dataset from file, shape: {df.shape}")
            return {
                "data": df.to_dict(orient='records'),
                "columns": list(df.columns),
                "metadata": {
                    "shape": df.shape,
                    "dtypes": df.dtypes.astype(str).to_dict()
                }
            }
        
        logger.warning(f"No file found in {settings.UPLOAD_DIR} for dataset {dataset_id}")
        
        # OPTION 2: Try to load metadata from local storage
        try:
            dataset_metadata = local_storage.load_json(f"dataset_{dataset_id}")
            logger.info(f"Loaded dataset metadata for {dataset_id}")
            
            # Check if metadata contains actual data
            if 'data' in dataset_metadata and dataset_metadata['data']:
                logger.info(f"Found data in metadata for dataset {dataset_id}")
                return {
                    "data": dataset_metadata['data'], 
                    "columns": dataset_metadata.get('columns', []),
                    "metadata": dataset_metadata.get('metadata', {})
                }
            
            # Try to find CSV file using metadata paths
            csv_filename = dataset_metadata.get('filename', f"{dataset_id}.csv")
            csv_paths = [
                settings.UPLOAD_DIR / csv_filename,
                settings.UPLOAD_DIR / dataset_metadata.get('original_filename', ''),
                Path(f"temp/{csv_filename}"),
                Path(f"temp/{dataset_metadata.get('original_filename', '')}"),
            ]
            
            for csv_path in csv_paths:
                if csv_path.exists():
                    logger.info(f"Found CSV at {csv_path}")
                    df = pd.read_csv(csv_path)
                    logger.info(f"Loaded CSV data, shape: {df.shape}")
                    return {
                        "data": df.to_dict(orient='records'),
                        "columns": list(df.columns),
                        "metadata": dataset_metadata.get('metadata', {})
                    }
                    
        except FileNotFoundError:
            logger.warning(f"Dataset metadata not found in local storage for {dataset_id}")
        except Exception as e:
            logger.error(f"Error loading from local storage: {e}")
        
        # OPTION 3: Try to load from database as last resort
        try:
            dataset = await db_manager.get_dataset(dataset_id)
            if dataset:
                logger.info(f"Loaded dataset from database for {dataset_id}")
                if isinstance(dataset, dict) and 'data' in dataset:
                    return dataset
                    
                # Convert to expected format
                df = pd.DataFrame(dataset)
                return {
                    "data": df.to_dict(orient='records'),
                    "columns": list(df.columns),
                    "metadata": {}
                }
        except Exception as e:
            logger.error(f"Error loading from database: {e}")
        
        # If we get here, dataset was not found
        logger.error(f"Dataset {dataset_id} not found in any location")
        logger.info(f"Searched locations:")
        logger.info(f"  - Upload directory: {settings.UPLOAD_DIR}")
        logger.info(f"  - Local storage: dataset_{dataset_id}.json")
        logger.info(f"  - Database")
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found or empty. Please upload the dataset again."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_id}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading dataset: {str(e)}"
        )
# Replace the _perform_analysis function in backend/app/api/ai_insights.py
# Around line 145

async def _perform_analysis(
    dataset_id: str,
    analysis_type: AnalysisType,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Helper function to perform analysis and return results"""
    context = context or {}
    
    try:
        # FIXED: Load dataset from actual file first, then fallback
        from pathlib import Path
        from app.core.config import settings
        from app.api.upload import load_dataset
        
        dataset_files = list(settings.UPLOAD_DIR.glob(f"{dataset_id}.*"))
        
        if dataset_files:
            # Load from actual file
            filepath = dataset_files[0]
            df = await load_dataset(filepath)
            logger.info(f"Loaded dataset from file for analysis, shape: {df.shape}")
        else:
            # Fallback to get_dataset method
            dataset = await get_dataset(dataset_id)
            df = pd.DataFrame(dataset['data'])
            logger.info(f"Loaded dataset from storage for analysis, shape: {df.shape}")
        
        # Perform the requested analysis
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
            
            # Simple correlation analysis
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
                "insights": insights,
                "correlations": top_correlations,
                "strong_correlations": strong_correlations,
                "correlation_matrix": corr_matrix.to_dict()
            }
            
        elif analysis_type == AnalysisType.MISSING_VALUES_ANALYSIS:
            result = await missing_values_processor.analyze_missing_patterns(df)
            
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
        import traceback
        logger.error(traceback.format_exc())
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

# Replace the get_correlation_insights function in backend/app/api/ai_insights.py
# Around line 268

@router.get("/{dataset_id}/correlations", response_model=Dict[str, Any])
async def get_correlation_insights(dataset_id: str):
    """Get AI-powered correlation insights with proper dataset loading"""
    try:
        # Load from actual file first (same pattern as missing-values endpoint)
        from pathlib import Path
        from app.core.config import settings
        from app.api.upload import load_dataset
        
        dataset_files = list(settings.UPLOAD_DIR.glob(f"{dataset_id}.*"))
        
        if dataset_files:
            # Load from actual file
            filepath = dataset_files[0]
            df = await load_dataset(filepath)
            logger.info(f"Loaded dataset from file for correlation insights, shape: {df.shape}")
        else:
            # Fallback to get_dataset method
            dataset = await get_dataset(dataset_id)
            df = pd.DataFrame(dataset['data'])
            logger.info(f"Loaded dataset from storage for correlation insights, shape: {df.shape}")
        
        # Check if we have enough numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return {
                "success": False,
                "analysis_type": "correlation_insights",
                "dataset_id": dataset_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "error": "At least two numeric columns are required for correlation analysis",
                    "numeric_columns_found": len(numeric_df.columns),
                    "total_columns": len(df.columns),
                    "insights": ["No numeric columns available for correlation analysis"]
                }
            }
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Extract correlations
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
        
        # Create insights
        strong_correlations = [c for c in top_correlations if abs(c["correlation"]) > 0.7]
        moderate_correlations = [c for c in top_correlations if 0.3 <= abs(c["correlation"]) <= 0.7]
        weak_correlations = [c for c in top_correlations if abs(c["correlation"]) < 0.3]
        
        insights = []
        insights.append(f"Analyzed {len(corr_matrix.columns)} numeric variables")
        
        if strong_correlations:
            insights.append(f"Found {len(strong_correlations)} strong correlations (|r| > 0.7)")
            for c in strong_correlations[:3]:
                direction = "positive" if c["correlation"] > 0 else "negative"
                insights.append(f"• Strong {direction} correlation: {c['var1']} ↔ {c['var2']} (r={c['correlation']})")
        
        if moderate_correlations:
            insights.append(f"Found {len(moderate_correlations)} moderate correlations (0.3 ≤ |r| ≤ 0.7)")
            for c in moderate_correlations[:2]:
                direction = "positive" if c["correlation"] > 0 else "negative"
                insights.append(f"• Moderate {direction} correlation: {c['var1']} ↔ {c['var2']} (r={c['correlation']})")
        
        if not strong_correlations and not moderate_correlations:
            insights.append("No strong correlations detected in the dataset")
            insights.append("Variables appear to be relatively independent")
        
        insights.append("💡 Strong correlations may indicate multicollinearity in predictive modeling")
        
        result = {
            "insights": insights,
            "correlations": top_correlations,
            "strong_correlations": strong_correlations,
            "moderate_correlations": moderate_correlations,
            "correlation_matrix": corr_matrix.to_dict(),
            "numeric_columns": list(corr_matrix.columns),
            "summary": {
                "total_pairs": len(correlations),
                "strong_count": len(strong_correlations),
                "moderate_count": len(moderate_correlations),
                "weak_count": len(weak_correlations)
            }
        }
        
        return {
            "success": True,
            "analysis_type": "correlation_insights",
            "dataset_id": dataset_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Error getting correlation insights for dataset {dataset_id}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting correlation insights: {str(e)}"
        )
@router.get("/{dataset_id}/missing-values", response_model=Dict[str, Any])
async def get_missing_values_insights(dataset_id: str):
    """Get comprehensive missing values analysis"""
    try:
        # Load from actual file first, then fallback to storage
        from pathlib import Path
        from app.core.config import settings
        from app.api.upload import load_dataset
        
        dataset_files = list(settings.UPLOAD_DIR.glob(f"{dataset_id}.*"))
        
        if dataset_files:
            filepath = dataset_files[0]
            df = await load_dataset(filepath)
            logger.info(f"Loaded dataset from file for missing values insights, shape: {df.shape}")
        else:
            dataset = await get_dataset(dataset_id)
            df = pd.DataFrame(dataset['data'])
            logger.info(f"Loaded dataset from storage for missing values insights, shape: {df.shape}")
        
        # Use the missing values processor for comprehensive analysis
        analysis = await missing_values_processor.analyze_missing_patterns(df)
        
        return {
            "success": True,
            "analysis_type": "missing_values",
            "dataset_id": dataset_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": analysis
        }
        
    except Exception as e:
        logger.error(f"Error getting missing values insights for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting missing values insights: {str(e)}"
        )

@router.get("/{dataset_id}/outliers", response_model=Dict[str, Any])
async def get_outlier_insights(dataset_id: str):
    """Get comprehensive outlier analysis"""
    try:
        dataset = await get_dataset(dataset_id)
        df = pd.DataFrame(dataset['data'])
        
        analysis = await outlier_processor.analyze_outlier_patterns(df)
        
        return {
            "success": True,
            "analysis_type": "outlier_analysis",
            "dataset_id": dataset_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": analysis
        }
        
    except Exception as e:
        logger.error(f"Error analyzing outliers for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing outliers: {str(e)}"
        )

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

# Missing Values Treatment Endpoints

@router.get("/{dataset_id}/missing-values/analysis", response_model=Dict[str, Any])
async def get_missing_values_analysis(dataset_id: str):
    """Get comprehensive missing values analysis"""
    try:
        # First try to load from actual file like statistics endpoint does
        from pathlib import Path
        from app.core.config import settings
        from app.api.upload import load_dataset
        
        dataset_files = list(settings.UPLOAD_DIR.glob(f"{dataset_id}.*"))
        
        if dataset_files:
            # Load from actual file
            filepath = dataset_files[0]
            df = await load_dataset(filepath)
            logger.info(f"Loaded dataset from file for missing values analysis, shape: {df.shape}")
        else:
            # Fallback to get_dataset method
            dataset = await get_dataset(dataset_id)
            df = pd.DataFrame(dataset['data'])
            logger.info(f"Loaded dataset from storage for missing values analysis, shape: {df.shape}")
        
        analysis = await missing_values_processor.analyze_missing_patterns(df)
        
        return {
            "success": True,
            "analysis_type": "missing_values_analysis",
            "dataset_id": dataset_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": analysis
        }
        
    except Exception as e:
        logger.error(f"Error analyzing missing values for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing missing values: {str(e)}"
        )

@router.post("/{dataset_id}/missing-values/treat")
async def treat_missing_values_fixed(
    dataset_id: str,
    method: str = Query(..., description="Treatment method to apply"),
    column: Optional[str] = Query(None, description="Target column (optional)"),
    n_neighbors: Optional[int] = Query(5, description="Number of neighbors for KNN"),
    constant_value: Optional[Any] = Query(0, description="Constant value for constant imputation"),
    threshold: Optional[float] = Query(0.5, description="Threshold for drop_columns method")
):
    """
    FIXED: Apply missing value treatment method and persist the changes
    Now properly handles all parameters and saves the treated dataset
    """
    try:
        from pathlib import Path
        from app.core.config import settings
        from app.api.upload import load_dataset
        
        # Load dataset from file or storage
        dataset_files = list(settings.UPLOAD_DIR.glob(f"{dataset_id}.*"))
        
        if dataset_files:
            filepath = dataset_files[0]
            df = await load_dataset(filepath)
            logger.info(f"✅ Loaded dataset from file: {filepath}, shape: {df.shape}")
        else:
            dataset = await get_dataset(dataset_id)
            df = pd.DataFrame(dataset['data'])
            logger.info(f"✅ Loaded dataset from storage, shape: {df.shape}")
        
        # Store original metrics
        original_shape = df.shape
        original_missing = df.isnull().sum().sum()
        
        logger.info(f"🔧 Applying treatment: {method}")
        logger.info(f"📊 Original shape: {original_shape}, Missing: {original_missing}")
        
        # Apply treatment using the processor
        df_treated, treatment_info = await missing_values_processor.treat_missing_values(
            df, method, column, 
            n_neighbors=n_neighbors,
            constant_value=constant_value,
            threshold=threshold
        )
        
        # Calculate final metrics
        final_missing = df_treated.isnull().sum().sum()
        
        # Update treatment info with comprehensive details
        treatment_info.update({
            "original_shape": list(original_shape),
            "final_shape": list(df_treated.shape),
            "original_missing": int(original_missing),
            "missing_values_remaining": int(final_missing),
            "rows_removed": int(original_shape[0] - df_treated.shape[0]),
            "columns_removed": int(original_shape[1] - df_treated.shape[1]),
            "method": method,
            "parameters": {
                "column": column,
                "n_neighbors": n_neighbors if method == "knn_imputation" else None,
                "constant_value": constant_value if method == "constant_imputation" else None,
                "threshold": threshold if method == "drop_columns" else None
            }
        })
        
        logger.info(f"✅ Treatment applied - Final shape: {df_treated.shape}, Missing: {final_missing}")
        
        # Save treated dataset
        treated_dataset_id = f"{dataset_id}_treated"
        treated_csv_path = settings.UPLOAD_DIR / f"{treated_dataset_id}.csv"
        df_treated.to_csv(treated_csv_path, index=False)
        logger.info(f"💾 Saved treated CSV: {treated_csv_path}")
        
        # Create metadata
        treated_metadata = {
            "id": treated_dataset_id,
            "original_id": dataset_id,
            "name": f"Dataset {dataset_id} - Cleaned",
            "filename": f"{treated_dataset_id}.csv",
            "data": df_treated.to_dict('records'),
            "columns": list(df_treated.columns),
            "metadata": {
                "original_dataset_id": dataset_id,
                "treatment_method": method,
                "treatment_info": treatment_info,
                "created_at": datetime.utcnow().isoformat(),
                "is_treated": True,
                "treatment_applied": True,
                "shape": list(df_treated.shape),
                "dtypes": {col: str(dtype) for col, dtype in df_treated.dtypes.items()},
                "csv_file_path": str(treated_csv_path),
                "missing_values_count": int(final_missing),
                "has_missing_values": bool(df_treated.isnull().any().any())
            }
        }
        
        # Save metadata
        local_storage.save_json(f"dataset_{treated_dataset_id}", treated_metadata)
        logger.info(f"💾 Saved treated metadata")
        
        # Update original dataset metadata
        try:
            original_metadata = local_storage.load_json(f"dataset_{dataset_id}")
            original_metadata['metadata'] = original_metadata.get('metadata', {})
            original_metadata['metadata']['treatment_applied'] = True
            original_metadata['metadata']['treated_dataset_id'] = treated_dataset_id
            original_metadata['metadata']['treatment_timestamp'] = datetime.utcnow().isoformat()
            local_storage.save_json(f"dataset_{dataset_id}", original_metadata)
            logger.info(f"✅ Updated original metadata")
        except Exception as e:
            logger.warning(f"⚠️ Could not update original metadata: {e}")
        
        # Save treatment history
        try:
            history = local_storage.load_json(f"treatment_history_{dataset_id}")
        except:
            history = {"treatments": []}
        
        history["treatments"].append({
            "method": method,
            "timestamp": datetime.utcnow().isoformat(),
            "treatment_info": treatment_info,
            "treated_dataset_id": treated_dataset_id,
            "parameters": treatment_info["parameters"]
        })
        
        local_storage.save_json(f"treatment_history_{dataset_id}", history)
        logger.info(f"💾 Saved treatment history")
        
        logger.info(f"🎉 Treatment completed successfully!")
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "treated_dataset_id": treated_dataset_id,
            "treatment_info": treatment_info,
            "timestamp": datetime.utcnow().isoformat(),
            "message": f"Treatment applied successfully. Dataset saved as {treated_dataset_id}",
            "csv_path": str(treated_csv_path)
        }
        
    except Exception as e:
        logger.error(f"❌ Error treating missing values: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error treating missing values: {str(e)}"
        )


@router.get("/{dataset_id}/treatment-status")
async def get_treatment_status(dataset_id: str):
    """
    Check if treatment has been applied to this dataset
    Returns status information for workflow control
    """
    try:
        from pathlib import Path
        from app.core.config import settings
        
        # Check if treated dataset exists
        treated_dataset_id = f"{dataset_id}_treated"
        treated_files = list(settings.UPLOAD_DIR.glob(f"{treated_dataset_id}.*"))
        
        has_treated_file = len(treated_files) > 0
        
        # Check metadata for treatment status
        try:
            original_metadata = local_storage.load_json(f"dataset_{dataset_id}")
            treatment_applied = original_metadata.get('metadata', {}).get('treatment_applied', False)
            treated_id = original_metadata.get('metadata', {}).get('treated_dataset_id')
        except:
            treatment_applied = False
            treated_id = None
        
        # Check if dataset has missing values
        try:
            dataset = await get_dataset(dataset_id)
            df = pd.DataFrame(dataset['data'])
            has_missing = bool(df.isnull().any().any())
            missing_count = int(df.isnull().sum().sum())
        except:
            has_missing = None
            missing_count = None
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "treatment_applied": treatment_applied or has_treated_file,
            "has_treated_file": has_treated_file,
            "treated_dataset_id": treated_id or treated_dataset_id if has_treated_file else None,
            "has_missing_values": has_missing,
            "missing_values_count": missing_count,
            "can_proceed_to_outliers": (treatment_applied or not has_missing),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error checking treatment status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking treatment status: {str(e)}"
        )


@router.get("/{dataset_id}/active-dataset")
async def get_active_dataset(dataset_id: str):
    """
    Get the active dataset (treated if available, otherwise original)
    This ensures downstream operations use the cleaned data
    """
    try:
        # Check if treated version exists
        treated_dataset_id = f"{dataset_id}_treated"
        
        try:
            treated_dataset = await get_dataset(treated_dataset_id)
            logger.info(f"✅ Using treated dataset: {treated_dataset_id}")
            return {
                "success": True,
                "dataset_id": treated_dataset_id,
                "is_treated": True,
                "original_id": dataset_id,
                "data": treated_dataset
            }
        except:
            # Fall back to original
            original_dataset = await get_dataset(dataset_id)
            logger.info(f"📊 Using original dataset: {dataset_id}")
            return {
                "success": True,
                "dataset_id": dataset_id,
                "is_treated": False,
                "data": original_dataset
            }
            
    except Exception as e:
        logger.error(f"Error getting active dataset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting active dataset: {str(e)}"
        )

@router.get("/{dataset_id}/missing-values/preview")
async def preview_missing_values_treatment(
    dataset_id: str,
    method: str,
    column: Optional[str] = None,
    n_neighbors: Optional[int] = 5,
    constant_value: Optional[Any] = 0,
    threshold: Optional[float] = 0.5
):
    """Preview what missing value treatment would do"""
    try:
        # Load from actual file first, then fallback to storage
        from pathlib import Path
        from app.core.config import settings
        from app.api.upload import load_dataset
        
        dataset_files = list(settings.UPLOAD_DIR.glob(f"{dataset_id}.*"))
        
        if dataset_files:
            filepath = dataset_files[0]
            df = await load_dataset(filepath)
            logger.info(f"Loaded dataset from file for preview, shape: {df.shape}")
        else:
            dataset = await get_dataset(dataset_id)
            df = pd.DataFrame(dataset['data'])
            logger.info(f"Loaded dataset from storage for preview, shape: {df.shape}")
        
        preview = await missing_values_processor.get_treatment_preview(
            df, method, column,
            n_neighbors=n_neighbors,
            constant_value=constant_value,
            threshold=threshold
        )
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "preview": preview,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error previewing missing values treatment for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error previewing treatment: {str(e)}"
        )

@router.get("/{dataset_id}/treatment-history")
async def get_treatment_history(dataset_id: str):
    """Get treatment history for a dataset"""
    try:
        history = local_storage.load_json(f"treatment_history_{dataset_id}")
        return {
            "success": True,
            "dataset_id": dataset_id,
            "history": history,
            "timestamp": datetime.utcnow().isoformat()
        }
    except FileNotFoundError:
        return {
            "success": True,
            "dataset_id": dataset_id,
            "history": {"treatments": []},
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting treatment history for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting treatment history: {str(e)}"
        )

@router.get("/{dataset_id}/download/csv")
async def download_dataset_csv(dataset_id: str):
    """Download dataset as CSV (original or treated)"""
    try:
        from fastapi.responses import FileResponse
        from pathlib import Path
        
        # First try to find the CSV file in upload directory
        dataset_files = list(settings.UPLOAD_DIR.glob(f"{dataset_id}.*"))
        
        if dataset_files:
            # Found CSV file, serve it directly
            csv_file = dataset_files[0]
            is_treated = "_treated" in dataset_id
            filename_suffix = "_cleaned" if is_treated else "_original"
            
            return FileResponse(
                path=csv_file,
                media_type="text/csv",
                filename=f"{dataset_id}{filename_suffix}.csv"
            )
        else:
            # Fallback to generating CSV from stored data
            from fastapi.responses import StreamingResponse
            import io
            
            dataset = await get_dataset(dataset_id)
            df = pd.DataFrame(dataset['data'])
            
            # Determine filename based on whether it's treated or original
            is_treated = dataset.get('metadata', {}).get('is_treated', False)
            filename_suffix = "_cleaned" if is_treated else "_original"
            
            # Create CSV in memory
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            # Create streaming response
            response = StreamingResponse(
                io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={dataset_id}{filename_suffix}.csv"}
            )
            
            return response
        
    except Exception as e:
        logger.error(f"Error downloading dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error downloading dataset: {str(e)}"
        )

# Outlier Detection and Treatment Endpoints

@router.get("/{dataset_id}/outliers/analysis", response_model=Dict[str, Any])
async def get_outlier_analysis(dataset_id: str):
    """Get comprehensive outlier analysis"""
    try:
        dataset = await get_dataset(dataset_id)
        df = pd.DataFrame(dataset['data'])
        
        analysis = await outlier_processor.analyze_outlier_patterns(df)
        
        return {
            "success": True,
            "analysis_type": "outlier_analysis",
            "dataset_id": dataset_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": analysis
        }
        
    except Exception as e:
        logger.error(f"Error analyzing outliers for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing outliers: {str(e)}"
        )

@router.post("/{dataset_id}/outliers/detect")
async def detect_outliers(
    dataset_id: str,
    method: str = "iqr",
    column: Optional[str] = None,
    threshold: Optional[float] = 3.0,
    multiplier: Optional[float] = 1.5,
    contamination: Optional[float] = 0.1,
    lower_percentile: Optional[float] = 1,
    upper_percentile: Optional[float] = 99
):
    """Detect outliers using specified method"""
    try:
        dataset = await get_dataset(dataset_id)
        df = pd.DataFrame(dataset['data'])
        
        # Apply detection
        detection_result = await outlier_processor.detect_outliers(
            df, method, column,
            threshold=threshold,
            multiplier=multiplier,
            contamination=contamination,
            lower=lower_percentile,
            upper=upper_percentile
        )
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "detection_result": detection_result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error detecting outliers for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error detecting outliers: {str(e)}"
        )

@router.post("/{dataset_id}/outliers/treat")
async def treat_outliers(
    dataset_id: str,
    method: str,
    column: Optional[str] = None,
    detection_method: Optional[str] = "iqr",
    z_threshold: Optional[float] = 3.0,
    lower_percentile: Optional[float] = 5,
    upper_percentile: Optional[float] = 95
):
    """Apply outlier treatment method"""
    try:
        dataset = await get_dataset(dataset_id)
        df = pd.DataFrame(dataset['data'])
        
        # Apply treatment
        df_treated, treatment_info = await outlier_processor.treat_outliers(
            df, method, column,
            detection_method=detection_method,
            z_threshold=z_threshold,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile
        )
        
        # Save treated dataset
        treated_dataset_id = f"{dataset_id}_outliers_treated_{method}"
        treated_data = {
            "id": treated_dataset_id,
            "name": f"{dataset.get('name', 'Dataset')} - {method} outlier treatment",
            "data": df_treated.to_dict('records'),
            "metadata": {
                "original_dataset_id": dataset_id,
                "treatment_method": method,
                "treatment_info": treatment_info,
                "created_at": datetime.utcnow().isoformat()
            }
        }
        
        # Store the treated dataset
        local_storage.save_json(f"dataset_{treated_dataset_id}", treated_data)
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "treated_dataset_id": treated_dataset_id,
            "treatment_info": treatment_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error treating outliers for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error treating outliers: {str(e)}"
        )

@router.get("/{dataset_id}/outliers/preview")
async def preview_outlier_treatment(
    dataset_id: str,
    method: str,
    column: Optional[str] = None,
    detection_method: Optional[str] = "iqr",
    z_threshold: Optional[float] = 3.0,
    lower_percentile: Optional[float] = 5,
    upper_percentile: Optional[float] = 95
):
    """Preview what outlier treatment would do"""
    try:
        dataset = await get_dataset(dataset_id)
        df = pd.DataFrame(dataset['data'])
        
        preview = await outlier_processor.get_treatment_preview(
            df, method, column,
            detection_method=detection_method,
            z_threshold=z_threshold,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile
        )
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "preview": preview,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error previewing outlier treatment for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error previewing treatment: {str(e)}"
        )

# Add this debug endpoint to backend/app/api/ai_insights.py
# Add it after the other router endpoints

@router.get("/debug/datasets")
async def debug_list_datasets():
    """Debug endpoint to list all available datasets and their locations"""
    try:
        from pathlib import Path
        from app.core.config import settings
        import os
        
        datasets_info = {
            "upload_directory": str(settings.UPLOAD_DIR),
            "files_in_upload_dir": [],
            "local_storage_files": [],
            "available_datasets": []
        }
        
        # List files in upload directory
        if settings.UPLOAD_DIR.exists():
            for filepath in settings.UPLOAD_DIR.glob("*"):
                if filepath.is_file():
                    datasets_info["files_in_upload_dir"].append({
                        "filename": filepath.name,
                        "dataset_id": filepath.stem,
                        "size": filepath.stat().st_size,
                        "extension": filepath.suffix
                    })
        
        # List local storage files
        local_storage_dir = Path("local_storage")
        if local_storage_dir.exists():
            for filepath in local_storage_dir.glob("dataset_*.json"):
                try:
                    metadata = local_storage.load_json(filepath.stem)
                    datasets_info["local_storage_files"].append({
                        "filename": filepath.name,
                        "dataset_id": metadata.get("dataset_id", "unknown"),
                        "has_data": "data" in metadata,
                        "columns": metadata.get("columns", [])
                    })
                except:
                    pass
        
        # List datasets from database
        try:
            db_datasets = await db_manager.list_datasets()
            datasets_info["database_datasets"] = db_datasets
        except Exception as e:
            datasets_info["database_error"] = str(e)
        
        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "datasets_info": datasets_info
        }
        
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


@router.get("/debug/dataset/{dataset_id}")
async def debug_dataset_info(dataset_id: str):
    """Debug endpoint to check if a specific dataset can be loaded"""
    try:
        from pathlib import Path
        from app.core.config import settings
        
        debug_info = {
            "dataset_id": dataset_id,
            "checks": []
        }
        
        # Check 1: File in upload directory
        dataset_files = list(settings.UPLOAD_DIR.glob(f"{dataset_id}.*"))
        debug_info["checks"].append({
            "location": "upload_directory",
            "found": len(dataset_files) > 0,
            "files": [str(f) for f in dataset_files]
        })
        
        # Check 2: Local storage metadata
        try:
            metadata = local_storage.load_json(f"dataset_{dataset_id}")
            debug_info["checks"].append({
                "location": "local_storage",
                "found": True,
                "has_data": "data" in metadata,
                "data_rows": len(metadata.get("data", [])) if "data" in metadata else 0,
                "columns": metadata.get("columns", [])
            })
        except:
            debug_info["checks"].append({
                "location": "local_storage",
                "found": False
            })
        
        # Check 3: Database
        try:
            db_dataset = await db_manager.get_dataset(dataset_id)
            debug_info["checks"].append({
                "location": "database",
                "found": db_dataset is not None
            })
        except Exception as e:
            debug_info["checks"].append({
                "location": "database",
                "found": False,
                "error": str(e)
            })
        
        # Check 4: Try to load using get_dataset
        try:
            dataset = await get_dataset(dataset_id)
            debug_info["load_test"] = {
                "success": True,
                "rows": len(dataset.get("data", [])),
                "columns": len(dataset.get("columns", []))
            }
        except Exception as e:
            debug_info["load_test"] = {
                "success": False,
                "error": str(e)
            }
        
        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "debug_info": debug_info
        }
        
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }