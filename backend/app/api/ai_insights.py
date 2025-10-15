"""
AI Insights API Endpoints - FINAL FIXED VERSION
ALL ISSUES RESOLVED
"""
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import re
from typing import Dict, Any, Optional
from datetime import datetime

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

# Database manager instance
db_manager = DatabaseManager()


async def load_dataset_from_file(dataset_id: str) -> pd.DataFrame:
    """Load dataset from actual CSV file - MOST RELIABLE METHOD"""
    try:
        # Look for CSV file in upload directory
        dataset_files = list(settings.UPLOAD_DIR.glob(f"{dataset_id}.*"))
        
        if not dataset_files:
            raise FileNotFoundError(f"No file found for dataset {dataset_id}")
        
        filepath = dataset_files[0]
        logger.info(f"✅ Loading dataset from file: {filepath}")
        
        # Read CSV file
        df = pd.read_csv(filepath)
        logger.info(f"✅ Loaded dataset, shape: {df.shape}, columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error loading dataset from file: {str(e)}")
        raise


async def get_dataset(dataset_id: str) -> dict:
    """Load dataset by ID - FIXED VERSION"""
    try:
        logger.info(f"Loading dataset {dataset_id}")
        
        # Try to load from actual uploaded file FIRST
        try:
            df = await load_dataset_from_file(dataset_id)
            return {
                "data": df.to_dict(orient='records'),
                "columns": list(df.columns),
                "metadata": {
                    "shape": df.shape,
                    "dtypes": df.dtypes.astype(str).to_dict()
                }
            }
        except FileNotFoundError:
            logger.warning(f"File not found for {dataset_id}, trying local storage")
        
        # Fallback to local storage
        try:
            dataset_metadata = local_storage.load_json(f"dataset_{dataset_id}")
            if 'data' in dataset_metadata and dataset_metadata['data']:
                logger.info(f"✅ Loaded from local storage")
                return {
                    "data": dataset_metadata['data'],
                    "columns": dataset_metadata.get('columns', []),
                    "metadata": dataset_metadata.get('metadata', {})
                }
        except Exception as e:
            logger.warning(f"Local storage failed: {e}")
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found. Please upload the dataset again."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading dataset: {str(e)}"
        )


# ============================================================================
# AI INSIGHTS ENDPOINTS - FIXED
# ============================================================================

@router.get("/{dataset_id}/overview")
async def get_dataset_overview(dataset_id: str):
    """Get AI-powered dataset overview insights - FIXED"""
    try:
        df = await load_dataset_from_file(dataset_id)
        
        # Generate overview
        overview = {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        }
        
        return {
            "success": True,
            "analysis_type": "dataset_overview",
            "dataset_id": dataset_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": overview
        }
        
    except Exception as e:
        logger.error(f"Error getting overview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/statistics")
async def get_statistical_insights(dataset_id: str):
    """Get AI-powered statistical insights - FIXED"""
    try:
        df = await load_dataset_from_file(dataset_id)
        
        # Generate statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        stats = {
            "numeric_summary": df[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {},
            "categorical_summary": {
                col: {
                    "unique_values": int(df[col].nunique()),
                    "most_common": str(df[col].mode().iloc[0]) if not df[col].mode().empty else None,
                    "frequency": int(df[col].value_counts().iloc[0]) if len(df[col]) > 0 else 0
                }
                for col in df.select_dtypes(include=['object']).columns
            }
        }
        
        return {
            "success": True,
            "analysis_type": "statistical_summary",
            "dataset_id": dataset_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MISSING VALUES ENDPOINTS - COMPLETELY FIXED
# ============================================================================

@router.get("/{dataset_id}/missing-values")
async def get_missing_values_insights(dataset_id: str):
    """Get comprehensive missing values analysis - FIXED"""
    try:
        logger.info(f"📊 Getting missing values insights for {dataset_id}")
        
        # Load dataset from file
        df = await load_dataset_from_file(dataset_id)
        
        # Analyze missing patterns
        analysis = await missing_values_processor.analyze_missing_patterns(df)
        
        return {
            "success": True,
            "analysis_type": "missing_values",
            "dataset_id": dataset_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": analysis
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting missing values insights: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting missing values insights: {str(e)}"
        )


@router.post("/{dataset_id}/missing-values/treat")
async def treat_missing_values(
    dataset_id: str,
    method: str = Query(..., description="Treatment method"),
    column: Optional[str] = Query(None, description="Target column (optional)"),
    n_neighbors: int = Query(5, description="KNN neighbors"),
    constant_value: str = Query("0", description="Constant value"),
    threshold: float = Query(0.5, description="Drop threshold")
):
    """Apply missing value treatment - COMPLETELY FIXED"""
    try:
        logger.info(f"🔧 Treating missing values for {dataset_id}")
        logger.info(f"Method: {method}, Column: {column}")
        
        # Load original dataset
        df_original = await load_dataset_from_file(dataset_id)
        original_shape = df_original.shape
        original_missing = int(df_original.isnull().sum().sum())
        
        logger.info(f"📊 Original: shape={original_shape}, missing={original_missing}")
        
        # Convert constant_value to appropriate type
        try:
            if constant_value.replace('.', '', 1).replace('-', '', 1).isdigit():
                constant_value_converted = float(constant_value) if '.' in constant_value else int(constant_value)
            else:
                constant_value_converted = constant_value
        except:
            constant_value_converted = 0
        
        # Apply treatment
        df_treated, treatment_info = await missing_values_processor.treat_missing_values(
            df_original,
            method=method,
            column=column,
            n_neighbors=n_neighbors,
            constant_value=constant_value_converted,
            threshold=threshold
        )
        
        final_shape = df_treated.shape
        final_missing = int(df_treated.isnull().sum().sum())
        
        logger.info(f"✅ Treated: shape={final_shape}, missing={final_missing}")
        
        # Save treated dataset
        treated_dataset_id = f"{dataset_id}_treated"
        treated_csv_path = settings.UPLOAD_DIR / f"{treated_dataset_id}.csv"
        
        # Save CSV file
        df_treated.to_csv(treated_csv_path, index=False)
        logger.info(f"💾 Saved CSV: {treated_csv_path}")
        
        # Create metadata
        treated_metadata = {
            "id": treated_dataset_id,
            "dataset_id": treated_dataset_id,
            "original_id": dataset_id,
            "name": f"Dataset {dataset_id} - Cleaned",
            "filename": f"{treated_dataset_id}.csv",
            "original_filename": f"{treated_dataset_id}.csv",
            "data": df_treated.to_dict('records'),
            "columns": list(df_treated.columns),
            "metadata": {
                "original_dataset_id": dataset_id,
                "treatment_method": method,
                "treatment_applied": True,
                "is_treated": True,
                "created_at": datetime.utcnow().isoformat(),
                "shape": list(final_shape),
                "dtypes": {col: str(dtype) for col, dtype in df_treated.dtypes.items()},
                "csv_file_path": str(treated_csv_path),
                "missing_values_count": final_missing,
                "has_missing_values": final_missing > 0,
                "original_shape": list(original_shape),
                "original_missing": original_missing,
                "rows_removed": int(original_shape[0] - final_shape[0]),
                "columns_removed": int(original_shape[1] - final_shape[1])
            }
        }
        
        # Save metadata
        local_storage.save_json(f"dataset_{treated_dataset_id}", treated_metadata)
        logger.info(f"💾 Saved metadata")
        
        # Update original dataset metadata
        try:
            original_metadata = local_storage.load_json(f"dataset_{dataset_id}")
            if 'metadata' not in original_metadata:
                original_metadata['metadata'] = {}
            original_metadata['metadata']['treatment_applied'] = True
            original_metadata['metadata']['treated_dataset_id'] = treated_dataset_id
            original_metadata['metadata']['treatment_timestamp'] = datetime.utcnow().isoformat()
            local_storage.save_json(f"dataset_{dataset_id}", original_metadata)
            logger.info(f"✅ Updated original metadata")
        except Exception as e:
            logger.warning(f"⚠️ Could not update original metadata: {e}")
        
        # Save treatment history - FIXED
        try:
            history_key = f"treatment_history_{dataset_id}"
            try:
                history = local_storage.load_json(history_key)
                if 'treatments' not in history:
                    history = {"treatments": []}
            except:
                history = {"treatments": []}
            
            history["treatments"].append({
                "method": method,
                "column": column,
                "timestamp": datetime.utcnow().isoformat(),
                "treated_dataset_id": treated_dataset_id,
                "original_shape": list(original_shape),
                "final_shape": list(final_shape),
                "missing_removed": original_missing - final_missing
            })
            
            local_storage.save_json(history_key, history)
            logger.info(f"💾 Saved treatment history")
        except Exception as e:
            logger.warning(f"⚠️ Could not save history: {e}")
        
        logger.info(f"🎉 Treatment completed successfully!")
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "treated_dataset_id": treated_dataset_id,
            "treatment_info": {
                "method": method,
                "column": column,
                "original_shape": list(original_shape),
                "final_shape": list(final_shape),
                "original_missing": original_missing,
                "missing_values_remaining": final_missing,
                "rows_removed": int(original_shape[0] - final_shape[0]),
                "columns_removed": int(original_shape[1] - final_shape[1]),
                "parameters": {
                    "n_neighbors": n_neighbors if method == "knn_imputation" else None,
                    "constant_value": constant_value if method == "constant_imputation" else None,
                    "threshold": threshold if method == "drop_columns" else None
                }
            },
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
    """Check if treatment has been applied - FIXED TO CHECK TREATED DATASET"""
    try:
        logger.info(f"🔍 Checking treatment status for {dataset_id}")
        
        # Check if treated file exists
        treated_dataset_id = f"{dataset_id}_treated"
        treated_files = list(settings.UPLOAD_DIR.glob(f"{treated_dataset_id}.*"))
        has_treated_file = len(treated_files) > 0
        
        logger.info(f"Treated file exists: {has_treated_file}")
        
        # Check metadata
        treatment_applied = False
        try:
            original_metadata = local_storage.load_json(f"dataset_{dataset_id}")
            treatment_applied = original_metadata.get('metadata', {}).get('treatment_applied', False)
            logger.info(f"Metadata treatment_applied: {treatment_applied}")
        except:
            pass
        
        # FIXED: Check the TREATED dataset for missing values if it exists
        has_missing = None
        missing_count = None
        
        if has_treated_file:
            # Check treated dataset
            try:
                df_treated = await load_dataset_from_file(treated_dataset_id)
                has_missing = bool(df_treated.isnull().any().any())
                missing_count = int(df_treated.isnull().sum().sum())
                logger.info(f"Treated dataset has {missing_count} missing values")
            except:
                pass
        else:
            # Check original dataset
            try:
                df = await load_dataset_from_file(dataset_id)
                has_missing = bool(df.isnull().any().any())
                missing_count = int(df.isnull().sum().sum())
                logger.info(f"Original dataset has {missing_count} missing values")
            except:
                pass
        
        # Determine if can proceed to outliers
        # Can proceed if: treatment applied AND treated dataset has no missing values
        # OR original dataset has no missing values
        can_proceed = False
        if has_treated_file and treatment_applied:
            # If treatment was applied, check if treated dataset is clean
            can_proceed = (has_missing is False) or (missing_count == 0)
        else:
            # If no treatment, can proceed only if original has no missing values
            can_proceed = (has_missing is False)
        
        logger.info(f"Can proceed to outliers: {can_proceed}")
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "treatment_applied": has_treated_file or treatment_applied,
            "has_treated_file": has_treated_file,
            "treated_dataset_id": treated_dataset_id if has_treated_file else None,
            "has_missing_values": has_missing,
            "missing_values_count": missing_count,
            "can_proceed_to_outliers": can_proceed,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error checking treatment status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking treatment status: {str(e)}"
        )


@router.get("/{dataset_id}/missing-values/preview")
async def preview_missing_values_treatment(
    dataset_id: str,
    method: str,
    column: Optional[str] = None,
    n_neighbors: int = 5,
    constant_value: str = "0",
    threshold: float = 0.5
):
    """Preview treatment effects - FIXED"""
    try:
        df = await load_dataset_from_file(dataset_id)
        
        # Convert constant_value
        try:
            if constant_value.replace('.', '', 1).replace('-', '', 1).isdigit():
                constant_value_converted = float(constant_value) if '.' in constant_value else int(constant_value)
            else:
                constant_value_converted = constant_value
        except:
            constant_value_converted = 0
        
        preview = await missing_values_processor.get_treatment_preview(
            df, method, column,
            n_neighbors=n_neighbors,
            constant_value=constant_value_converted,
            threshold=threshold
        )
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "preview": preview,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error previewing treatment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error previewing treatment: {str(e)}"
        )


@router.get("/{dataset_id}/download/csv")
async def download_dataset_csv(dataset_id: str):
    """Download dataset as CSV - FIXED"""
    try:
        # Find CSV file
        dataset_files = list(settings.UPLOAD_DIR.glob(f"{dataset_id}.*"))
        
        if dataset_files:
            csv_file = dataset_files[0]
            is_treated = "_treated" in dataset_id
            filename_suffix = "_cleaned" if is_treated else "_original"
            
            return FileResponse(
                path=csv_file,
                media_type="text/csv",
                filename=f"{dataset_id}{filename_suffix}.csv",
                headers={"Content-Disposition": f"attachment; filename={dataset_id}{filename_suffix}.csv"}
            )
        else:
            # Generate from stored data
            import io
            dataset = await get_dataset(dataset_id)
            df = pd.DataFrame(dataset['data'])
            
            is_treated = dataset.get('metadata', {}).get('is_treated', False)
            filename_suffix = "_cleaned" if is_treated else "_original"
            
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            return StreamingResponse(
                io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={dataset_id}{filename_suffix}.csv"}
            )
        
    except Exception as e:
        logger.error(f"❌ Error downloading dataset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error downloading dataset: {str(e)}"
        )


# ============================================================================
# OTHER ENDPOINTS
# ============================================================================

@router.get("/{dataset_id}/correlations")
async def get_correlation_insights(dataset_id: str):
    """Get correlation insights - FIXED"""
    try:
        df = await load_dataset_from_file(dataset_id)
        
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return {
                "success": False,
                "error": "At least two numeric columns required"
            }
        
        corr_matrix = numeric_df.corr()
        
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
        
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        return {
            "success": True,
            "analysis_type": "correlations",
            "dataset_id": dataset_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "correlations": correlations[:10],
                "correlation_matrix": corr_matrix.to_dict()
            }
        }
    except Exception as e:
        logger.error(f"Error getting correlations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/outliers")
async def get_outlier_insights(dataset_id: str):
    """Get outlier analysis - FIXED"""
    try:
        df = await load_dataset_from_file(dataset_id)
        analysis = await outlier_processor.analyze_outlier_patterns(df)
        
        return {
            "success": True,
            "analysis_type": "outliers",
            "dataset_id": dataset_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": analysis
        }
    except Exception as e:
        logger.error(f"Error getting outliers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{dataset_id}/outliers/treat")
async def treat_outliers(
    dataset_id: str,
    method: str = Query(..., description="Treatment method"),
    column: Optional[str] = Query(None, description="Target column (optional)"),
    detection_method: str = Query("iqr", description="Detection method for outlier removal/replacement"),
    z_threshold: float = Query(3.0, description="Z-score threshold"),
    lower_percentile: float = Query(1.0, description="Lower percentile for capping/winsorizing"),
    upper_percentile: float = Query(99.0, description="Upper percentile for capping/winsorizing")
):
    """Apply outlier treatment to the dataset"""
    try:
        logger.info(f"🔧 Treating outliers for {dataset_id}")
        logger.info(f"Method: {method}, Column: {column}")
        df_original = await load_dataset_from_file(dataset_id)
        original_shape = df_original.shape

        # Prepare kwargs for treatment
        treatment_kwargs = {
            "detection_method": detection_method,
            "z_threshold": z_threshold,
            "lower_percentile": lower_percentile,
            "upper_percentile": upper_percentile
        }
        # Remove unused keys for methods that don't use them
        if method not in ["cap_percentile", "winsorize"]:
            treatment_kwargs.pop("lower_percentile", None)
            treatment_kwargs.pop("upper_percentile", None)
        if method != "cap_z_score":
            treatment_kwargs.pop("z_threshold", None)
        if method not in ["remove", "replace_median", "replace_mean"]:
            treatment_kwargs.pop("detection_method", None)

        df_treated, treatment_info = await outlier_processor.treat_outliers(
            df_original,
            method=method,
            column=column,
            **treatment_kwargs
        )
        final_shape = df_treated.shape

        # Save treated dataset
        treated_dataset_id = f"{dataset_id}_treated"
        treated_csv_path = settings.UPLOAD_DIR / f"{treated_dataset_id}.csv"
        df_treated.to_csv(treated_csv_path, index=False)
        logger.info(f"💾 Saved CSV: {treated_csv_path}")

        # Create metadata
        treated_metadata = {
            "id": treated_dataset_id,
            "dataset_id": treated_dataset_id,
            "original_id": dataset_id,
            "name": f"Dataset {dataset_id} - Outliers Treated",
            "filename": f"{treated_dataset_id}.csv",
            "original_filename": f"{treated_dataset_id}.csv",
            "data": df_treated.to_dict('records'),
            "columns": list(df_treated.columns),
            "metadata": {
                "original_dataset_id": dataset_id,
                "treatment_method": method,
                "treatment_applied": True,
                "is_treated": True,
                "created_at": datetime.utcnow().isoformat(),
                "shape": list(final_shape),
                "dtypes": {col: str(dtype) for col, dtype in df_treated.dtypes.items()},
                "csv_file_path": str(treated_csv_path),
                "original_shape": list(original_shape),
                "rows_removed": int(original_shape[0] - final_shape[0]),
            }
        }
        local_storage.save_json(f"dataset_{treated_dataset_id}", treated_metadata)
        logger.info(f"💾 Saved metadata for treated dataset")

        return {
            "success": True,
            "dataset_id": dataset_id,
            "treated_dataset_id": treated_dataset_id,
            "treatment_info": treatment_info,
            "timestamp": datetime.utcnow().isoformat(),
            "message": f"Outlier treatment applied successfully. Dataset saved as {treated_dataset_id}",
            "csv_path": str(treated_csv_path)
        }
    except Exception as e:
        logger.error(f"❌ Error treating outliers: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error treating outliers: {str(e)}"
        )


@router.get("/{dataset_id}/outliers/preview")
async def preview_outlier_treatment(
    dataset_id: str,
    method: str,
    column: Optional[str] = None,
    detection_method: str = Query("iqr", description="Detection method for outlier removal/replacement"),
    z_threshold: float = Query(3.0, description="Z-score threshold"),
    lower_percentile: float = Query(1.0, description="Lower percentile for capping/winsorizing"),
    upper_percentile: float = Query(99.0, description="Upper percentile for capping/winsorizing")
):
    """Preview outlier treatment effects"""
    try:
        df = await load_dataset_from_file(dataset_id)
        treatment_kwargs = {
            "detection_method": detection_method,
            "z_threshold": z_threshold,
            "lower_percentile": lower_percentile,
            "upper_percentile": upper_percentile
        }
        # Remove unused keys for methods that don't use them
        if method not in ["cap_percentile", "winsorize"]:
            treatment_kwargs.pop("lower_percentile", None)
            treatment_kwargs.pop("upper_percentile", None)
        if method != "cap_z_score":
            treatment_kwargs.pop("z_threshold", None)
        if method not in ["remove", "replace_median", "replace_mean"]:
            treatment_kwargs.pop("detection_method", None)

        preview = await outlier_processor.get_treatment_preview(
            df, method, column,
            **treatment_kwargs
        )
        return {
            "success": True,
            "dataset_id": dataset_id,
            "preview": preview,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error previewing outlier treatment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error previewing outlier treatment: {str(e)}"
        )

@router.get("/{dataset_id}/outliers/analysis")
async def get_outlier_analysis(dataset_id: str):
    """Get comprehensive outlier analysis using multiple methods"""
    try:
        logger.info(f"📊 Getting outlier analysis for {dataset_id}")
        
        df = await load_dataset_from_file(dataset_id)
        analysis = await outlier_processor.analyze_outlier_patterns(df)
        
        return {
            "success": True,
            "analysis_type": "outlier_analysis",
            "dataset_id": dataset_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": analysis
        }
        
    except Exception as e:
        logger.error(f"❌ Error in outlier analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/outliers/detect")
async def detect_outliers(
    dataset_id: str,
    method: str = Query("z_score"),
    column: Optional[str] = Query(None),
    threshold: float = Query(3.0),
    multiplier: float = Query(1.5),
    lower_percentile: int = Query(1),
    upper_percentile: int = Query(99)
):
    """Detect outliers using specified method"""
    try:
        logger.info(f"🔍 Detecting outliers for {dataset_id} using {method}")
        
        df = await load_dataset_from_file(dataset_id)
        
        params = {
            "threshold": threshold,
            "multiplier": multiplier,
            "lower": lower_percentile,
            "upper": upper_percentile
        }
        
        result = await outlier_processor.detect_outliers(
            df,
            method=method,
            column=column,
            **params
        )
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "method": method,
            "timestamp": datetime.utcnow().isoformat(),
            "data": result
        }
        
    except Exception as e:
        logger.error(f"❌ Error detecting outliers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/outliers/summary")
async def get_outlier_summary(dataset_id: str):
    """Get summary of outliers in dataset"""
    try:
        logger.info(f"📈 Getting outlier summary for {dataset_id}")
        
        df = await load_dataset_from_file(dataset_id)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return {
                "success": False,
                "error": "No numeric columns found",
                "dataset_id": dataset_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        summary = {
            "dataset_id": dataset_id,
            "total_rows": len(df),
            "numeric_columns": len(numeric_cols),
            "columns_analysis": {},
            "overall_statistics": {
                "columns_with_outliers": 0,
                "total_potential_outliers": 0,
                "dataset_outlier_percentage": 0.0
            }
        }
        
        total_outliers = 0
        cols_with_outliers = 0
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
            
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (col_data < lower_bound) | (col_data > upper_bound)
            outlier_count = outliers_mask.sum()
            
            if outlier_count > 0:
                cols_with_outliers += 1
                total_outliers += outlier_count
            
            outlier_values = col_data[outliers_mask].tolist() if outlier_count > 0 else []
            
            summary["columns_analysis"][col] = {
                "outlier_count": int(outlier_count),
                "outlier_percentage": float((outlier_count / len(col_data)) * 100),
                "bounds": {
                    "lower": float(lower_bound),
                    "upper": float(upper_bound)
                },
                "statistics": {
                    "min": float(col_data.min()),
                    "q1": float(Q1),
                    "median": float(col_data.median()),
                    "q3": float(Q3),
                    "max": float(col_data.max()),
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std())
                },
                "sample_outliers": outlier_values[:10]
            }
        
        summary["overall_statistics"]["columns_with_outliers"] = cols_with_outliers
        summary["overall_statistics"]["total_potential_outliers"] = total_outliers
        summary["overall_statistics"]["dataset_outlier_percentage"] = float((total_outliers / len(df)) * 100)
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": summary
        }
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/outliers/preview-treatment")
async def preview_treatment(
    dataset_id: str,
    method: str = Query(...),
    column: Optional[str] = Query(None),
    detection_method: str = Query("iqr"),
    lower_percentile: int = Query(1),
    upper_percentile: int = Query(99),
    z_threshold: float = Query(3.0)
):
    """Preview outlier treatment"""
    try:
        df = await load_dataset_from_file(dataset_id)
        
        params = {
            "detection_method": detection_method,
            "lower_percentile": lower_percentile,
            "upper_percentile": upper_percentile,
            "z_threshold": z_threshold
        }
        
        preview = await outlier_processor.get_treatment_preview(
            df,
            method=method,
            column=column,
            **params
        )
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "method": method,
            "timestamp": datetime.utcnow().isoformat(),
            "data": preview
        }
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{dataset_id}/outliers/treat")
async def treat_outliers(
    dataset_id: str,
    method: str = Query(...),
    column: Optional[str] = Query(None),
    detection_method: str = Query("iqr"),
    lower_percentile: int = Query(1),
    upper_percentile: int = Query(99),
    z_threshold: float = Query(3.0)
):
    """Apply outlier treatment"""
    try:
        logger.info(f"🔧 Treating outliers for {dataset_id}")
        
        df_original = await load_dataset_from_file(dataset_id)
        original_shape = df_original.shape
        
        params = {
            "detection_method": detection_method,
            "lower_percentile": lower_percentile,
            "upper_percentile": upper_percentile,
            "z_threshold": z_threshold
        }
        
        df_treated, treatment_info = await outlier_processor.treat_outliers(
            df_original,
            method=method,
            column=column,
            **params
        )
        
        final_shape = df_treated.shape
        
        treated_dataset_id = f"{dataset_id}_outliers_treated"
        treated_csv_path = settings.UPLOAD_DIR / f"{treated_dataset_id}.csv"
        
        df_treated.to_csv(treated_csv_path, index=False)
        logger.info(f"💾 Saved CSV: {treated_csv_path}")
        
        treated_metadata = {
            "id": treated_dataset_id,
            "dataset_id": treated_dataset_id,
            "original_id": dataset_id,
            "name": f"Dataset {dataset_id} - Outliers Treated",
            "filename": f"{treated_dataset_id}.csv",
            "data": df_treated.to_dict('records'),
            "columns": list(df_treated.columns),
            "metadata": {
                "original_dataset_id": dataset_id,
                "treatment_method": method,
                "treatment_applied": True,
                "is_treated": True,
                "treatment_type": "outlier",
                "created_at": datetime.utcnow().isoformat(),
                "shape": list(final_shape),
                "dtypes": {col: str(dtype) for col, dtype in df_treated.dtypes.items()},
                "csv_file_path": str(treated_csv_path),
                "original_shape": list(original_shape),
                "rows_removed": int(original_shape[0] - final_shape[0]),
                "columns_removed": int(original_shape[1] - final_shape[1]),
                "treatment_details": treatment_info
            }
        }
        
        local_storage.save_json(f"dataset_{treated_dataset_id}", treated_metadata)
        
        try:
            original_metadata = local_storage.load_json(f"dataset_{dataset_id}")
            if 'metadata' not in original_metadata:
                original_metadata['metadata'] = {}
            original_metadata['metadata']['outlier_treatment_applied'] = True
            original_metadata['metadata']['treated_dataset_id'] = treated_dataset_id
            original_metadata['metadata']['treatment_timestamp'] = datetime.utcnow().isoformat()
            local_storage.save_json(f"dataset_{dataset_id}", original_metadata)
        except Exception as e:
            logger.warning(f"⚠️ Could not update original metadata: {e}")
        
        try:
            history_key = f"treatment_history_{dataset_id}"
            try:
                history = local_storage.load_json(history_key)
                if 'treatments' not in history:
                    history = {"treatments": []}
            except:
                history = {"treatments": []}
            
            history["treatments"].append({
                "type": "outlier",
                "method": method,
                "column": column,
                "timestamp": datetime.utcnow().isoformat(),
                "treated_dataset_id": treated_dataset_id,
                "original_shape": list(original_shape),
                "final_shape": list(final_shape),
                "rows_removed": int(original_shape[0] - final_shape[0])
            })
            
            local_storage.save_json(history_key, history)
        except Exception as e:
            logger.warning(f"⚠️ Could not save history: {e}")
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "treated_dataset_id": treated_dataset_id,
            "treatment_info": {
                "method": method,
                "column": column,
                "original_shape": list(original_shape),
                "final_shape": list(final_shape),
                "rows_removed": int(original_shape[0] - final_shape[0]),
                "columns_removed": int(original_shape[1] - final_shape[1]),
                "details": treatment_info
            },
            "timestamp": datetime.utcnow().isoformat(),
            "message": f"Outlier treatment applied. Dataset saved as {treated_dataset_id}",
            "csv_path": str(treated_csv_path)
        }
        
    except Exception as e:
        logger.error(f"❌ Error treating outliers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/outliers/treatment-status")
async def get_treatment_status(dataset_id: str):
    """Check if outlier treatment has been applied"""
    try:
        treated_dataset_id = f"{dataset_id}_outliers_treated"
        treated_files = list(settings.UPLOAD_DIR.glob(f"{treated_dataset_id}.*"))
        has_treated_file = len(treated_files) > 0
        
        treatment_applied = False
        try:
            original_metadata = local_storage.load_json(f"dataset_{dataset_id}")
            treatment_applied = original_metadata.get('metadata', {}).get('outlier_treatment_applied', False)
        except:
            pass
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "treatment_applied": has_treated_file or treatment_applied,
            "has_treated_file": has_treated_file,
            "treated_dataset_id": treated_dataset_id if has_treated_file else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/outliers/treatment-history")
async def get_treatment_history(dataset_id: str):
    """Get history of outlier treatments applied"""
    try:
        history_key = f"treatment_history_{dataset_id}"
        
        try:
            history = local_storage.load_json(history_key)
        except:
            history = {"treatments": []}
        
        outlier_treatments = [
            t for t in history.get("treatments", [])
            if t.get("type") == "outlier"
        ]
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "treatments": outlier_treatments,
            "total_treatments": len(outlier_treatments),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))