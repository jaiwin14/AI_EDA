"""
Preprocessing API Endpoints
Split from preprocessor logic; mirrors EDA router patterns
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional, List
import pandas as pd
from pathlib import Path
import logging

from app.core.config import settings
from app.core.database import local_storage
from app.api.upload import load_dataset
from app.ml.preprocessor import IntelligentPreprocessor
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/preprocessing", tags=["preprocessing"])


def _resolve_dataset_path(dataset_id: str, source: str = "auto") -> Path:
    source = (source or "auto").lower()
    if source not in {"auto", "original", "outlier_treated"}:
        raise HTTPException(status_code=400, detail="Invalid source. Use auto|original|outlier_treated")
    if source in {"auto", "outlier_treated"}:
        treated_candidates = list(settings.UPLOAD_DIR.glob(f"{dataset_id}_outlier_treated.*"))
        if treated_candidates:
            return treated_candidates[0]
        if source == "outlier_treated":
            raise HTTPException(status_code=404, detail="Outlier-treated dataset not found")
    base_candidates = list(settings.UPLOAD_DIR.glob(f"{dataset_id}.*"))
    if base_candidates:
        return base_candidates[0]
    raise HTTPException(status_code=404, detail="Dataset not found")


def _preprocessor_from_settings(use_gemini_query: Optional[bool] = None) -> IntelligentPreprocessor:
    # Enable Gemini only when API key exists and caller hasn't disabled it explicitly
    gemini_available = bool(settings.GEMINI_API_KEY)
    use_gemini = gemini_available if use_gemini_query is None else (gemini_available and use_gemini_query)
    return IntelligentPreprocessor(use_gemini=use_gemini)


@router.post("/{dataset_id}/preprocess")
async def preprocess_dataset(
    dataset_id: str,
    source: Optional[str] = Query(default="auto", description="auto|original|outlier_treated"),
    target_col: Optional[str] = Query(default=None),
    task_type: Optional[str] = Query(default=None),
    use_gemini: Optional[bool] = Query(default=None)
) -> Dict[str, Any]:
    try:
        filepath = _resolve_dataset_path(dataset_id, source)
        df = await load_dataset(filepath)
        pp = _preprocessor_from_settings(use_gemini)
        df_processed, steps = pp.preprocess(df, target_col=target_col, task_type=task_type)
        return {
            "dataset_id": dataset_id,
            "shape_before": list(df.shape),
            "shape_after": list(df_processed.shape),
            "steps": steps
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Preprocess failed for {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")


@router.post("/{dataset_id}/encode")
async def encode_features(
    dataset_id: str,
    source: Optional[str] = Query(default="auto", description="auto|original|outlier_treated"),
    target_col: Optional[str] = Query(default=None),
    task_type: Optional[str] = Query(default=None),
    use_gemini: Optional[bool] = Query(default=None)
) -> Dict[str, Any]:
    try:
        filepath = _resolve_dataset_path(dataset_id, source)
        df = await load_dataset(filepath)
        pp = _preprocessor_from_settings(use_gemini)
        column_types = pp.detect_column_types(df)
        df_out, steps = pp.encode_features(df, column_types, target_col=target_col, task_type=task_type)
        return {
            "dataset_id": dataset_id,
            "shape_before": list(df.shape),
            "shape_after": list(df_out.shape),
            "encoding": steps
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Encoding failed for {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Encoding failed: {str(e)}")


@router.post("/{dataset_id}/scale")
async def scale_numeric(
    dataset_id: str,
    source: Optional[str] = Query(default="auto", description="auto|original|outlier_treated"),
    target_col: Optional[str] = Query(default=None),
    use_gemini: Optional[bool] = Query(default=None)
) -> Dict[str, Any]:
    try:
        filepath = _resolve_dataset_path(dataset_id, source)
        df = await load_dataset(filepath)
        pp = _preprocessor_from_settings(use_gemini)
        column_types = pp.detect_column_types(df)
        df_out, steps = pp.scale_features(df, column_types, target_col=target_col)
        return {
            "dataset_id": dataset_id,
            "shape_before": list(df.shape),
            "shape_after": list(df_out.shape),
            "scaling": steps
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scaling failed for {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Scaling failed: {str(e)}")


@router.post("/{dataset_id}/reduce")
async def reduce_dimensions(
    dataset_id: str,
    source: Optional[str] = Query(default="auto", description="auto|original|outlier_treated"),
    target_col: Optional[str] = Query(default=None),
    use_gemini: Optional[bool] = Query(default=None)
) -> Dict[str, Any]:
    try:
        filepath = _resolve_dataset_path(dataset_id, source)
        df = await load_dataset(filepath)
        pp = _preprocessor_from_settings(use_gemini)
        column_types = pp.detect_column_types(df)
        df_out, steps = pp.reduce_dimensionality(df, column_types, target_col=target_col)
        return {
            "dataset_id": dataset_id,
            "shape_before": list(df.shape),
            "shape_after": list(df_out.shape),
            "reduction": steps
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reduction failed for {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Dimensionality reduction failed: {str(e)}")


@router.post("/{dataset_id}/preprocess/save")
async def preprocess_and_save(
    dataset_id: str,
    source: Optional[str] = Query(default="auto", description="auto|original|outlier_treated"),
    target_col: Optional[str] = Query(default=None),
    task_type: Optional[str] = Query(default=None),
    use_gemini: Optional[bool] = Query(default=None)
) -> Dict[str, Any]:
    """Run preprocessing, save CSV + steps JSON, and return a download link."""
    try:
        filepath = _resolve_dataset_path(dataset_id, source)
        df = await load_dataset(filepath)
        pp = _preprocessor_from_settings(use_gemini)
        df_processed, steps = pp.preprocess(df, target_col=target_col, task_type=task_type)

        # Save processed CSV
        out_name = f"{dataset_id}_preprocessed.csv"
        out_path = settings.UPLOAD_DIR / out_name
        df_processed.to_csv(out_path, index=False)

        # Save steps metadata
        meta_key = f"preprocess_{dataset_id}_steps"
        local_storage.save_json(meta_key, {
            "dataset_id": dataset_id,
            "source": source,
            "shape_before": list(df.shape),
            "shape_after": list(df_processed.shape),
            "steps": steps,
            "file": str(out_path)
        })

        return {
            "dataset_id": dataset_id,
            "shape_before": list(df.shape),
            "shape_after": list(df_processed.shape),
            "steps": steps,
            "filename": out_name,
            "download_url": f"/api/v1/preprocessing/{dataset_id}/download"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Preprocess save failed for {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocess save failed: {str(e)}")


@router.get("/{dataset_id}/download")
async def download_preprocessed(dataset_id: str):
    """Download the last saved preprocessed CSV for this dataset."""
    try:
        out_path = settings.UPLOAD_DIR / f"{dataset_id}_preprocessed.csv"
        if not out_path.exists():
            raise HTTPException(status_code=404, detail="Preprocessed file not found. Run preprocess/save first.")
        return FileResponse(path=out_path, filename=out_path.name, media_type='text/csv')
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed for {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to download preprocessed file")


