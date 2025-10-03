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

# Initialize processors
eda_processor = EDAProcessor()
missing_values_processor = MissingValuesProcessor()
outlier_processor = OutlierProcessor()

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
                return {"data": df.to_dict(orient='records'), "columns": list(df.columns), "metadata": dataset_metadata}
            else:
                logger.warning(f"No CSV data found for dataset {dataset_id}")
                logger.info(f"Searched paths: {csv_paths}")
                
                # Check if we have data stored in the metadata itself
                if 'data' in dataset_metadata:
                    logger.info(f"Found data in metadata for dataset {dataset_id}")
                    return {
                        "data": dataset_metadata['data'], 
                        "columns": dataset_metadata.get('columns', []),
                        "metadata": dataset_metadata
                    }
                
                # Generate synthetic data based on metadata for demonstration
                columns = dataset_metadata.get('metadata', {}).get('columns', [])
                dtypes = dataset_metadata.get('metadata', {}).get('dtypes', {})
                shape = dataset_metadata.get('metadata', {}).get('shape', [0, 0])
                
                if len(columns) > 0 and shape[0] > 0:
                    # Use the actual shape from metadata, not limited to 100
                    actual_rows = shape[0]
                    logger.info(f"Generating synthetic data with {actual_rows} rows and {len(columns)} columns")
                    
                    # Create synthetic data based on column types
                    synthetic_data = []
                    for i in range(actual_rows):
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
                    
                    return {"data": synthetic_data, "columns": columns, "metadata": dataset_metadata}
                else:
                    return {"data": [], "columns": columns, "metadata": dataset_metadata}
                
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
                "insights": insights,
                "correlations": top_correlations,
                "strong_correlations": strong_correlations,
                "correlation_matrix": corr_matrix.to_dict()
            }
        elif analysis_type == AnalysisType.MISSING_VALUES_ANALYSIS:
            # Use the advanced missing values processor
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
async def treat_missing_values(
    dataset_id: str,
    method: str,
    column: Optional[str] = None,
    n_neighbors: Optional[int] = 5,
    constant_value: Optional[Any] = 0,
    threshold: Optional[float] = 0.5
):
    """Apply missing value treatment method"""
    try:
        # Load from actual file first, then fallback to storage
        from pathlib import Path
        from app.core.config import settings
        from app.api.upload import load_dataset
        
        dataset_files = list(settings.UPLOAD_DIR.glob(f"{dataset_id}.*"))
        
        if dataset_files:
            filepath = dataset_files[0]
            df = await load_dataset(filepath)
            logger.info(f"Loaded dataset from file for treatment, shape: {df.shape}")
            dataset = {"data": df.to_dict('records'), "columns": list(df.columns)}
        else:
            dataset = await get_dataset(dataset_id)
            df = pd.DataFrame(dataset['data'])
            logger.info(f"Loaded dataset from storage for treatment, shape: {df.shape}")
        
        # Apply treatment
        df_treated, treatment_info = await missing_values_processor.treat_missing_values(
            df, method, column, 
            n_neighbors=n_neighbors,
            constant_value=constant_value,
            threshold=threshold
        )
        
        # Save treated dataset with consistent naming
        treated_dataset_id = f"{dataset_id}_treated"
        
        # Save treated dataset as CSV file in upload directory
        treated_csv_path = settings.UPLOAD_DIR / f"{treated_dataset_id}.csv"
        df_treated.to_csv(treated_csv_path, index=False)
        logger.info(f"Saved treated dataset CSV to: {treated_csv_path}")
        
        treated_data = {
            "id": treated_dataset_id,
            "name": f"{dataset.get('name', 'Dataset')} - Cleaned",
            "data": df_treated.to_dict('records'),
            "columns": list(df_treated.columns),
            "metadata": {
                "original_dataset_id": dataset_id,
                "treatment_method": method,
                "treatment_info": treatment_info,
                "created_at": datetime.utcnow().isoformat(),
                "is_treated": True,
                "shape": df_treated.shape,
                "dtypes": df_treated.dtypes.to_dict(),
                "csv_file_path": str(treated_csv_path)
            }
        }
        
        # Store the treated dataset metadata
        local_storage.save_json(f"dataset_{treated_dataset_id}", treated_data)
        
        # Also save treatment history
        try:
            history = local_storage.load_json(f"treatment_history_{dataset_id}")
        except:
            history = {"treatments": []}
        
        history["treatments"].append({
            "method": method,
            "timestamp": datetime.utcnow().isoformat(),
            "treatment_info": treatment_info,
            "treated_dataset_id": treated_dataset_id
        })
        
        local_storage.save_json(f"treatment_history_{dataset_id}", history)
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "treated_dataset_id": treated_dataset_id,
            "treatment_info": treatment_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error treating missing values for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error treating missing values: {str(e)}"
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