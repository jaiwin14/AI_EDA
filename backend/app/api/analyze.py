from fastapi import APIRouter, HTTPException, UploadFile, Depends
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import io
import json
from pydantic import BaseModel, Field
from datetime import datetime

from app.ml.preprocessor import IntelligentPreprocessor

router = APIRouter(prefix="/api/analyze", tags=["dataset-analysis"])

class ColumnAnalysis(BaseModel):
    name: str
    dtype: str
    unique_values: int
    missing_values: int
    missing_percentage: float
    is_numeric: bool
    is_categorical: bool
    is_datetime: bool
    is_binary: bool
    is_potential_target: bool
    potential_task_type: Optional[str] = None
    sample_values: List[Any]
    
    class Config:
        json_encoders = {
            np.ndarray: lambda v: v.tolist(),
            np.generic: lambda v: v.item(),
            pd.Timestamp: lambda v: v.isoformat(),
        }

class PreprocessingStep(BaseModel):
    step_name: str
    description: str
    parameters: Optional[Dict[str, Any]] = None
    columns_affected: List[str] = []
    
class PreprocessingReport(BaseModel):
    steps: List[PreprocessingStep]
    preprocessed_columns: List[str]
    memory_usage_before_mb: float
    memory_usage_after_mb: float
    
class DatasetAnalysis(BaseModel):
    filename: str
    total_rows: int
    total_columns: int
    columns: List[ColumnAnalysis]
    suggested_target: Optional[str] = None
    suggested_task_type: Optional[str] = None
    memory_usage_mb: float
    preprocessing_report: Optional[PreprocessingReport] = None
    
    class Config:
        json_encoders = {
            np.ndarray: lambda v: v.tolist(),
            np.generic: lambda v: v.item(),
            pd.Timestamp: lambda v: v.isoformat(),
        }

class DatasetAnalysis(BaseModel):
    filename: str
    total_rows: int
    total_columns: int
    columns: List[ColumnAnalysis]
    suggested_target: Optional[str] = None
    suggested_task_type: Optional[str] = None
    memory_usage_mb: float

def _read_uploaded_file(file: UploadFile) -> pd.DataFrame:
    """Read an uploaded file into a pandas DataFrame."""
    contents = file.file.read()
    
    try:
        if file.filename.endswith('.csv'):
            return pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xls', '.xlsx')):
            return pd.read_excel(io.BytesIO(contents))
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")

def _analyze_columns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Analyze columns of a DataFrame."""
    columns_analysis = []
    
    for col in df.columns:
        col_data = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(col_data)
        is_datetime = pd.api.types.is_datetime64_any_dtype(col_data)
        unique_vals = col_data.nunique()
        
        # Determine if column is categorical
        is_categorical = (not is_numeric and not is_datetime) or \
                        (unique_vals / len(col_data) < 0.05 and unique_vals < 100)
        
        # Determine if column is binary
        is_binary = unique_vals == 2
        
        # Check if column could be a target
        is_potential_target = False
        task_type = None
        
        if is_binary and not is_datetime:
            is_potential_target = True
            task_type = "binary_classification"
        elif is_categorical and 2 < unique_vals < 100 and not is_datetime:
            is_potential_target = True
            task_type = "multiclass_classification"
        elif is_numeric and unique_vals > 10 and not is_datetime:
            is_potential_target = True
            task_type = "regression"
        
        # Get sample values (handling non-serializable types)
        try:
            sample_values = col_data.dropna().head(5).tolist()
            # Convert numpy types to native Python types for JSON serialization
            sample_values = [
                val.item() if hasattr(val, 'item') and not isinstance(val, (str, bytes)) 
                else val for val in sample_values
            ]
        except Exception:
            sample_values = []
        
        # Add column analysis
        col_analysis = {
            'name': col,
            'dtype': str(col_data.dtype),
            'unique_values': int(unique_vals),
            'missing_values': int(col_data.isna().sum()),
            'missing_percentage': float(col_data.isna().mean() * 100),
            'is_numeric': is_numeric,
            'is_categorical': is_categorical,
            'is_datetime': is_datetime,
            'is_binary': is_binary,
            'is_potential_target': is_potential_target,
            'potential_task_type': task_type,
            'sample_values': sample_values
        }
        columns_analysis.append(col_analysis)
    
    return columns_analysis

def _suggest_target(columns_analysis: List[Dict[str, Any]]) -> tuple[Optional[str], Optional[str]]:
    """Suggest the best target column and task type."""
    potential_targets = [col for col in columns_analysis if col['is_potential_target']]
    
    # Prefer binary classification, then multiclass, then regression
    target_preference = ['binary_classification', 'multiclass_classification', 'regression']
    
    for task in target_preference:
        matching_targets = [t for t in potential_targets if t['potential_task_type'] == task]
        if matching_targets:
            # Prefer columns with no missing values
            best_target = min(matching_targets, key=lambda x: x['missing_percentage'])
            return best_target['name'], task
    
    return None, None

def _create_preprocessing_report(
    preprocessing_steps: Dict[str, Any],
    df_before: pd.DataFrame,
    df_after: pd.DataFrame
) -> Dict[str, Any]:
    """Create a preprocessing report from the preprocessing steps."""
    steps = []
    
    # Convert preprocessing steps to a list of PreprocessingStep objects
    if 'missing_values' in preprocessing_steps:
        missing_steps = preprocessing_steps['missing_values']
        for col, step in missing_steps.items():
            if step.get('action') == 'impute':
                steps.append({
                    'step_name': 'Missing Value Imputation',
                    'description': f"Imputed missing values in column '{col}' using {step.get('method')}",
                    'parameters': {'method': step.get('method')},
                    'columns_affected': [col]
                })
            elif step.get('action') == 'drop':
                steps.append({
                    'step_name': 'Column Removal',
                    'description': f"Dropped column '{col}' due to {step.get('reason', 'unspecified reason')}",
                    'columns_affected': [col]
                })
    
    if 'encoding' in preprocessing_steps:
        encoding_steps = preprocessing_steps['encoding']
        for col, step in encoding_steps.items():
            if 'method' in step:
                if step['method'] == 'onehot':
                    steps.append({
                        'step_name': 'One-Hot Encoding',
                        'description': f"Applied one-hot encoding to column '{col}'",
                        'columns_affected': step.get('columns', [col])
                    })
                elif step['method'] == 'label':
                    steps.append({
                        'step_name': 'Label Encoding',
                        'description': f"Applied label encoding to column '{col}'",
                        'columns_affected': [col]
                    })
    
    if 'scaling' in preprocessing_steps:
        scaling = preprocessing_steps['scaling']
        steps.append({
            'step_name': 'Feature Scaling',
            'description': f"Applied {scaling['type']} scaling to numerical features",
            'parameters': scaling.get('params', {}),
            'columns_affected': scaling.get('columns', [])
        })
    
    if 'dimensionality_reduction' in preprocessing_steps:
        reduction = preprocessing_steps['dimensionality_reduction']
        steps.append({
            'step_name': 'Dimensionality Reduction',
            'description': f"Applied {reduction['method'].upper()} for dimensionality reduction",
            'parameters': {
                'n_components': reduction['n_components'],
                'original_features': len(reduction['original_columns']),
                'reduced_features': len(reduction['new_columns'])
            },
            'columns_affected': reduction['new_columns']
        })
    
    return {
        'steps': steps,
        'preprocessed_columns': list(df_after.columns),
        'memory_usage_before_mb': df_before.memory_usage(deep=True).sum() / (1024 * 1024),
        'memory_usage_after_mb': df_after.memory_usage(deep=True).sum() / (1024 * 1024)
    }

@router.post("/", response_model=DatasetAnalysis)
async def analyze_dataset(
    file: UploadFile,
    preprocess_data: bool = True,
    use_gemini: bool = True
):
    """
    Analyze a dataset and optionally preprocess it.
    
    Args:
        file: The dataset file to analyze (CSV or Excel)
        preprocess_data: Whether to apply intelligent preprocessing
        use_gemini: Whether to use Gemini AI for decision making
        
    Returns:
        DatasetAnalysis with analysis results and optional preprocessing report
    """
    try:
        # Read the uploaded file
        df = _read_uploaded_file(file)
        
        # Store original DataFrame for comparison
        df_original = df.copy()
        
        # Initialize analysis results
        columns_analysis = _analyze_columns(df)
        suggested_target, suggested_task_type = _suggest_target(columns_analysis)
        
        # Initialize response
        analysis = {
            'filename': file.filename,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'columns': columns_analysis,
            'suggested_target': suggested_target,
            'suggested_task_type': suggested_task_type,
            'preprocessing_report': None
        }
        
        # Apply preprocessing if requested
        if preprocess_data and suggested_target is not None:
            preprocessor = IntelligentPreprocessor(use_gemini=use_gemini)
            
            # Apply preprocessing
            df_processed, preprocessing_steps = preprocessor.preprocess(
                df,
                target_col=suggested_target,
                task_type=suggested_task_type
            )
            
            # Create preprocessing report
            preprocessing_report = _create_preprocessing_report(
                preprocessing_steps,
                df_original,
                df_processed
            )
            
            analysis['preprocessing_report'] = preprocessing_report
            
            # Update column analysis for the processed data
            analysis['columns'] = _analyze_columns(df_processed)
            
            # Update memory usage
            analysis['memory_usage_mb'] = preprocessing_report['memory_usage_after_mb']
        
        return analysis
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )
