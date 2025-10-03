"""
EDA API Endpoints
Automated exploratory data analysis endpoints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

from app.core.config import settings
from app.utils.json_utils import serialize_for_json
from app.core.database import db_manager, local_storage
from app.api.upload import load_dataset
from ..ml.eda_processor import EDAProcessor
from ..ml.visualization_generator import VisualizationGenerator
from ..ml.ai_agent import ai_agent

logger = logging.getLogger(__name__)
router = APIRouter()

def _calculate_monotonicity(series):
    """Calculate monotonicity of a series"""
    if len(series) < 2:
        return "Not enough data"
    
    diff = series.diff().dropna()
    if len(diff) == 0:
        return "Constant"
    
    increasing = (diff >= 0).sum()
    decreasing = (diff <= 0).sum()
    total = len(diff)
    
    if increasing == total:
        return "Strictly Increasing"
    elif decreasing == total:
        return "Strictly Decreasing"
    elif increasing / total > 0.95:
        return "Mostly Increasing"
    elif decreasing / total > 0.95:
        return "Mostly Decreasing"
    else:
        return "Not Monotonic"

@router.post("/eda/{dataset_id}")
async def run_eda_analysis(
    dataset_id: str,
    background_tasks: BackgroundTasks,
    analysis_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run comprehensive EDA analysis on a dataset
    
    Args:
        dataset_id: ID of the uploaded dataset
        analysis_types: Optional list of specific analyses to run
                       Options: ['basic', 'missing', 'correlations', 'outliers', 'distributions']
    """
    
    # Find dataset file
    dataset_files = list(settings.UPLOAD_DIR.glob(f"{dataset_id}.*"))
    if not dataset_files:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found"
        )
    
    filepath = dataset_files[0]
    
    try:
        # Load dataset
        df = await load_dataset(filepath)
        
        # Initialize EDA processor
        eda_processor = EDAProcessor()
        viz_generator = VisualizationGenerator()
        
        # Determine which analyses to run
        if analysis_types is None:
            analysis_types = ['basic', 'missing', 'correlations', 'outliers', 'distributions']
        
        results = {
            "dataset_id": dataset_id,
            "analysis_timestamp": pd.Timestamp.now().isoformat(),
            "dataset_shape": df.shape,
            "analyses": {}
        }
        
        # Run requested analyses
        if 'basic' in analysis_types:
            logger.info(f"Running basic statistics for dataset {dataset_id}")
            basic_stats = await eda_processor.generate_basic_statistics(df)
            results["analyses"]["basic_statistics"] = basic_stats
            
            # Save to database
            await db_manager.save_eda_results(dataset_id, "basic_statistics", basic_stats)
        
        if 'missing' in analysis_types:
            logger.info(f"Running missing values analysis for dataset {dataset_id}")
            missing_analysis = await eda_processor.analyze_missing_values(df)
            results["analyses"]["missing_values"] = missing_analysis
            
            # Generate missing values visualization
            missing_viz = await viz_generator.create_missing_values_plot(df)
            results["analyses"]["missing_values"]["visualization"] = missing_viz
            
            await db_manager.save_eda_results(dataset_id, "missing_values", missing_analysis)
        
        if 'correlations' in analysis_types:
            logger.info(f"Running correlation analysis for dataset {dataset_id}")
            correlation_analysis = await eda_processor.analyze_correlations(df)
            results["analyses"]["correlations"] = correlation_analysis
            
            # Generate correlation heatmap
            corr_viz = await viz_generator.create_correlation_heatmap(df)
            results["analyses"]["correlations"]["visualization"] = corr_viz
            
            await db_manager.save_eda_results(dataset_id, "correlations", correlation_analysis)
        
        if 'outliers' in analysis_types:
            logger.info(f"Running outlier detection for dataset {dataset_id}")
            outlier_analysis = await eda_processor.detect_outliers(df)
            results["analyses"]["outliers"] = outlier_analysis
            
            # Generate outlier visualizations
            outlier_viz = await viz_generator.create_outlier_plots(df)
            results["analyses"]["outliers"]["visualizations"] = outlier_viz
            
            await db_manager.save_eda_results(dataset_id, "outliers", outlier_analysis)
        
        if 'distributions' in analysis_types:
            logger.info(f"Running distribution analysis for dataset {dataset_id}")
            dist_analysis = await eda_processor.analyze_distributions(df)
            results["analyses"]["distributions"] = dist_analysis
            
            # Generate distribution plots
            dist_viz = await viz_generator.create_distribution_plots(df)
            results["analyses"]["distributions"]["visualizations"] = dist_viz
            
            await db_manager.save_eda_results(dataset_id, "distributions", dist_analysis)
        
        # Schedule comprehensive report generation in background
        background_tasks.add_task(
            generate_comprehensive_report,
            dataset_id,
            df,
            results
        )
        
        return {
            "status": "success",
            "message": "EDA analysis completed successfully",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"EDA analysis failed for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"EDA analysis failed: {str(e)}"
        )

@router.get("/eda/{dataset_id}/results")
async def get_eda_results(
    dataset_id: str,
    analysis_type: Optional[str] = None
) -> Dict[str, Any]:
    """Get EDA results for a dataset"""
    
    try:
        # First try to load the actual dataset file to ensure consistency
        dataset_files = list(settings.UPLOAD_DIR.glob(f"{dataset_id}.*"))
        
        if dataset_files:
            # Load from actual file and generate fresh results
            filepath = dataset_files[0]
            df = await load_dataset(filepath)
            logger.info(f"Loaded dataset from file for EDA results, shape: {df.shape}")
            
            # Generate fresh basic statistics to ensure consistency
            eda_processor = EDAProcessor()
            basic_stats = await eda_processor.generate_basic_statistics(df)
            
            return {
                "dataset_id": dataset_id,
                "analyses": {
                    "basic_statistics": basic_stats,
                    "missing_values": None,
                    "correlations": None,
                    "distributions": None,
                    "outliers": None
                },
                "visualizations": [],
                "summary": {
                    "total_analyses": 1,
                    "completed_at": pd.Timestamp.now().isoformat(),
                    "status": "completed"
                }
            }
        
        # Fallback: Get results from database
        results = await db_manager.get_eda_results(dataset_id, analysis_type)
        
        if not results:
            # If no results found, return empty structure instead of 404
            logger.warning(f"No EDA results found for dataset {dataset_id}")
            return {
                "dataset_id": dataset_id,
                "analyses": {
                    "basic_statistics": None,
                    "missing_values": None,
                    "correlations": None,
                    "distributions": None,
                    "outliers": None
                },
                "visualizations": [],
                "summary": {
                    "total_analyses": 0,
                    "completed_at": None,
                    "status": "no_results"
                }
            }
        
        # Structure the results properly for frontend consumption
        analyses = {}
        visualizations = []
        
        for result in results:
            analysis_type_key = result.get("type", "unknown")
            analysis_data = result.get("results", {})
            
            # Map analysis types to expected structure
            if analysis_type_key in ["basic_statistics", "missing_values", "correlations", "distributions", "outliers"]:
                analyses[analysis_type_key] = analysis_data
            elif analysis_type_key == "visualizations":
                visualizations.extend(analysis_data if isinstance(analysis_data, list) else [analysis_data])
        
        # Get the latest timestamp
        latest_timestamp = None
        if results:
            timestamps = [result.get("created_at") for result in results if result.get("created_at")]
            if timestamps:
                latest_timestamp = max(timestamps)
        
        return {
            "dataset_id": dataset_id,
            "analyses": analyses,
            "visualizations": visualizations,
            "summary": {
                "total_analyses": len(results),
                "completed_at": latest_timestamp,
                "status": "completed" if results else "no_results"
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving EDA results for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve EDA results"
        )

@router.get("/eda/{dataset_id}/summary")
async def get_eda_summary(dataset_id: str) -> Dict[str, Any]:
    """Get a summary of all EDA analyses for a dataset"""
    
    try:
        # Get all EDA results
        all_results = await db_manager.get_eda_results(dataset_id)
        
        if not all_results:
            raise HTTPException(
                status_code=404,
                detail="No EDA results found for this dataset"
            )
        
        # Create summary
        summary = {
            "dataset_id": dataset_id,
            "total_analyses": len(all_results),
            "available_analyses": [result.get("analysis_type") for result in all_results],
            "last_updated": max([result.get("created_at", "") for result in all_results]),
            "key_insights": []
        }
        
        # Extract key insights from each analysis
        for result in all_results:
            analysis_type = result.get("analysis_type")
            analysis_data = result.get("results", {})
            
            if analysis_type == "basic_statistics":
                overview = analysis_data.get("overview", {})
                summary["key_insights"].append({
                    "type": "dataset_overview",
                    "insight": f"Dataset has {overview.get('total_rows', 0):,} rows and {overview.get('total_columns', 0)} columns"
                })
                
                if overview.get("duplicate_rows", 0) > 0:
                    summary["key_insights"].append({
                        "type": "data_quality",
                        "insight": f"Found {overview.get('duplicate_rows', 0):,} duplicate rows"
                    })
            
            elif analysis_type == "missing_values":
                missing_pct = analysis_data.get("overall_missing_percentage", 0)
                if missing_pct > 5:
                    summary["key_insights"].append({
                        "type": "data_quality",
                        "insight": f"Dataset has {missing_pct:.1f}% missing values"
                    })
            
            elif analysis_type == "correlations":
                strong_corrs = analysis_data.get("strong_correlations", {}).get("pearson", [])
                if strong_corrs:
                    summary["key_insights"].append({
                        "type": "relationships",
                        "insight": f"Found {len(strong_corrs)} strong correlations between features"
                    })
            
            elif analysis_type == "outliers":
                outlier_summary = analysis_data.get("outlier_summary", {})
                high_outlier_cols = [col for col, info in outlier_summary.items() 
                                   if info.get("severity") == "high"]
                if high_outlier_cols:
                    summary["key_insights"].append({
                        "type": "data_quality",
                        "insight": f"High outlier concentration in columns: {', '.join(high_outlier_cols[:3])}"
                    })
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating EDA summary for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate EDA summary"
        )

@router.get("/eda/{dataset_id}/statistics")
async def get_statistical_analysis(dataset_id: str) -> Dict[str, Any]:
    """
    Get comprehensive statistical analysis for the dataset
    
    Args:
        dataset_id: ID of the dataset
    """
    
    # Find dataset file
    dataset_files = list(settings.UPLOAD_DIR.glob(f"{dataset_id}.*"))
    if not dataset_files:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found"
        )
    
    filepath = dataset_files[0]
    
    try:
        # Load dataset
        df = await load_dataset(filepath)
        
        # Initialize EDA processor
        eda_processor = EDAProcessor()
        
        # Get comprehensive statistical analysis
        basic_stats = await eda_processor.generate_basic_statistics(df)
        missing_analysis = await eda_processor.analyze_missing_values(df)
        univariate_analysis = await eda_processor.univariate_analysis(df)
        data_types_analysis = await eda_processor.analyze_data_types(df)
        
        # Calculate comprehensive dataset overview
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Calculate missing values
        total_missing = df.isnull().sum().sum()
        total_cells = len(df) * len(df.columns)
        missing_percentage = (total_missing / total_cells) * 100 if total_cells > 0 else 0
        
        # Calculate duplicate rows
        duplicate_rows = df.duplicated().sum()
        duplicate_percentage = (duplicate_rows / len(df)) * 100 if len(df) > 0 else 0
        
        # Memory usage calculations
        memory_usage_bytes = df.memory_usage(deep=True).sum()
        memory_usage_mb = memory_usage_bytes / (1024 * 1024)
        average_record_size_bytes = memory_usage_bytes / len(df) if len(df) > 0 else 0
        
        # Generate comprehensive analysis for numeric columns
        detailed_numeric_analysis = {}
        if len(numeric_cols) > 0:
            describe_df = df[numeric_cols].describe()
            for col in numeric_cols:
                col_data = df[col].dropna()
                col_series = df[col]
                
                # Basic information
                distinct_count = col_series.nunique()
                missing_count = col_series.isnull().sum()
                total_count = len(col_series)
                non_null_count = col_series.count()
                
                # Calculate percentages
                distinct_percentage = (distinct_count / total_count) * 100 if total_count > 0 else 0
                missing_percentage = (missing_count / total_count) * 100 if total_count > 0 else 0
                
                # Check for infinite values
                infinite_count = 0
                infinite_percentage = 0
                if len(col_data) > 0:
                    infinite_count = int(np.isinf(col_data).sum())
                    infinite_percentage = (infinite_count / total_count) * 100 if total_count > 0 else 0
                
                # Count zeros and negatives
                zeros_count = int((col_data == 0).sum()) if len(col_data) > 0 else 0
                zeros_percentage = (zeros_count / total_count) * 100 if total_count > 0 else 0
                negative_count = int((col_data < 0).sum()) if len(col_data) > 0 else 0
                negative_percentage = (negative_count / total_count) * 100 if total_count > 0 else 0
                
                # Memory size for this column
                col_memory_bytes = col_series.memory_usage(deep=True)
                
                # Basic statistics
                if len(col_data) > 0:
                    min_val = float(col_data.min())
                    max_val = float(col_data.max())
                    mean_val = float(col_data.mean())
                    median_val = float(col_data.median())
                    std_val = float(col_data.std()) if len(col_data) > 1 else 0.0
                else:
                    min_val = max_val = mean_val = median_val = std_val = None
                
                # Quantile statistics
                quantile_stats = {}
                if len(col_data) > 0:
                    quantile_stats = {
                        "minimum": float(col_data.min()),
                        "percentile_5": float(col_data.quantile(0.05)),
                        "q1": float(col_data.quantile(0.25)),
                        "median": float(col_data.median()),
                        "q3": float(col_data.quantile(0.75)),
                        "percentile_95": float(col_data.quantile(0.95)),
                        "maximum": float(col_data.max()),
                        "range": float(col_data.max() - col_data.min()),
                        "iqr": float(col_data.quantile(0.75) - col_data.quantile(0.25))
                    }
                
                # Descriptive statistics
                descriptive_stats = {}
                if len(col_data) > 1:
                    cv = (std_val / mean_val * 100) if mean_val != 0 else 0
                    # Calculate Median Absolute Deviation manually since mad() is deprecated
                    mad = float((col_data - col_data.median()).abs().median()) if len(col_data) > 0 else 0
                    
                    descriptive_stats = {
                        "standard_deviation": std_val,
                        "coefficient_of_variation": round(cv, 2),
                        "kurtosis": float(col_data.kurtosis()),
                        "mean": mean_val,
                        "median_absolute_deviation": mad,
                        "skewness": float(col_data.skew()),
                        "sum": float(col_data.sum()),
                        "variance": float(col_data.var()),
                        "monotonicity": _calculate_monotonicity(col_data)
                    }
                
                # Common values (most frequent)
                common_values = {}
                if len(col_data) > 0:
                    value_counts = col_data.value_counts().head(10)
                    common_values = {
                        "values": [{"value": float(val), "count": int(count), "percentage": round((count/len(col_data))*100, 2)} 
                                 for val, count in value_counts.items()]
                    }
                
                # Extreme values
                extreme_values = {}
                if len(col_data) > 0:
                    sorted_data = col_data.sort_values()
                    extreme_values = {
                        "lowest": [{"value": float(val), "index": int(idx)} 
                                 for idx, val in sorted_data.head(5).items()],
                        "highest": [{"value": float(val), "index": int(idx)} 
                                  for idx, val in sorted_data.tail(5).items()]
                    }

                detailed_numeric_analysis[col] = {
                    # Basic information
                    "variable_name": col,
                    "data_type": str(col_series.dtype),
                    "is_uniform": distinct_count == 1,
                    "is_unique": distinct_count == total_count,
                    "distinct_count": distinct_count,
                    "distinct_percentage": round(distinct_percentage, 2),
                    "missing_count": missing_count,
                    "missing_percentage": round(missing_percentage, 2),
                    "infinite_count": infinite_count,
                    "infinite_percentage": round(infinite_percentage, 2),
                    "minimum": min_val,
                    "maximum": max_val,
                    "zeros_count": zeros_count,
                    "zeros_percentage": round(zeros_percentage, 2),
                    "negative_count": negative_count,
                    "negative_percentage": round(negative_percentage, 2),
                    "memory_size_bytes": int(col_memory_bytes),
                    
                    # Detailed analysis tabs
                    "quantile_statistics": quantile_stats,
                    "descriptive_statistics": descriptive_stats,
                    "common_values": common_values,
                    "extreme_values": extreme_values,
                    
                    # Basic statistics for backward compatibility
                    "count": float(describe_df.loc['count', col]),
                    "mean": float(describe_df.loc['mean', col]),
                    "std": float(describe_df.loc['std', col]),
                    "min": float(describe_df.loc['min', col]),
                    "25%": float(describe_df.loc['25%', col]),
                    "50%": float(describe_df.loc['50%', col]),
                    "75%": float(describe_df.loc['75%', col]),
                    "max": float(describe_df.loc['max', col])
                }

        # Generate detailed categorical summary
        categorical_describe = {}
        for col in categorical_cols:
            col_data = df[col].dropna()  # Remove NaN for length calculations
            value_counts = df[col].value_counts()
            total_count = len(df)
            missing_count = int(df[col].isnull().sum())
            distinct_count = int(df[col].nunique())
            
            # Basic info
            basic_info = {
                "variable_name": col,
                "data_type": str(df[col].dtype),
                "distinct": distinct_count,
                "distinct_percentage": round((distinct_count / total_count) * 100, 2),
                "unique_ratio": round(distinct_count / total_count, 4) if total_count > 0 else 0,
                "missing": missing_count,
                "missing_percentage": round((missing_count / total_count) * 100, 2),
                "memory_size": round(df[col].memory_usage(deep=True) / 1024, 2),  # in KB
                "most_frequent": str(value_counts.index[0]) if len(value_counts) > 0 else "N/A",
                "frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
            }
            
            # Length statistics (for string columns)
            length_stats = {}
            if col_data.dtype == 'object' and len(col_data) > 0:
                lengths = col_data.astype(str).str.len()
                length_stats = {
                    "max_length": int(lengths.max()) if len(lengths) > 0 else 0,
                    "median_length": float(lengths.median()) if len(lengths) > 0 else 0,
                    "mean_length": round(float(lengths.mean()), 2) if len(lengths) > 0 else 0,
                    "min_length": int(lengths.min()) if len(lengths) > 0 else 0
                }
            
            # Categories with frequency
            categories_data = []
            if len(value_counts) > 0:
                for category, count in value_counts.items():
                    categories_data.append({
                        "category": str(category),
                        "count": int(count),
                        "frequency_percent": round((count / total_count) * 100, 2)
                    })
            
            categorical_describe[col] = {
                # Basic statistics for backward compatibility
                "count": int(df[col].count()),
                "unique": distinct_count,
                "top": str(value_counts.index[0]) if len(value_counts) > 0 else "",
                "freq": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "missing": missing_count,
                "missing_percentage": round((missing_count / total_count) * 100, 2),
                
                # Detailed analysis
                "basic_info": basic_info,
                "length_stats": length_stats,
                "categories": categories_data
            }

        # Organize results with proper overview data
        results = {
            "dataset_id": dataset_id,
            "dataset_shape": [int(df.shape[0]), int(df.shape[1])],
            "analysis_timestamp": pd.Timestamp.now().isoformat(),
            "dataset_overview": {
                "total_rows": int(len(df)),
                "total_columns": int(len(df.columns)),
                "total_variables": int(len(df.columns)),  # Number of variables/attributes
                "number_of_data_records": int(len(df)),  # Number of data records
                "numeric_columns": int(len(numeric_cols)),
                "categorical_columns": int(len(categorical_cols)),
                "datetime_columns": int(len(datetime_cols)),
                "total_missing_cells": int(total_missing),
                "missing_cells_percentage": float(missing_percentage),
                "duplicate_rows": int(duplicate_rows),
                "duplicate_rows_percentage": float(duplicate_percentage),
                "total_memory_usage_mb": float(memory_usage_mb),
                "total_memory_usage_bytes": int(memory_usage_bytes),
                "average_record_size_bytes": float(average_record_size_bytes),
                "variable_types": {
                    "numeric": int(len(numeric_cols)),
                    "categorical": int(len(categorical_cols)),
                    "datetime": int(len(datetime_cols))
                }
            },
            "basic_statistics": {
                "numeric_summary": {
                    "describe": detailed_numeric_analysis
                },
                "categorical_summary": categorical_describe,
                "overview": {
                    "total_rows": int(len(df)),
                    "total_columns": int(len(df.columns)),
                    "numeric_columns": int(len(numeric_cols)),
                    "categorical_columns": int(len(categorical_cols))
                }
            },
            "missing_values_analysis": missing_analysis,
            "univariate_analysis": univariate_analysis,
            "data_types_analysis": data_types_analysis,
            "column_summary": []
        }
        
        # Create detailed column summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in df.columns:
            col_summary = {
                "column_name": col,
                "data_type": str(df[col].dtype),
                "column_type": "numerical" if col in numeric_cols else "categorical",
                "non_null_count": int(df[col].count()),
                "null_count": int(df[col].isnull().sum()),
                "null_percentage": float((df[col].isnull().sum() / len(df)) * 100),
                "unique_count": int(df[col].nunique()),
                "unique_ratio": float(df[col].nunique() / len(df))
            }
            
            if col in numeric_cols:
                # Calculate statistics only on non-null values
                col_data = df[col].dropna()
                
                if len(col_data) > 0:
                    # 5-point summary
                    min_val = float(col_data.min())
                    q25_val = float(col_data.quantile(0.25))
                    median_val = float(col_data.median())
                    q75_val = float(col_data.quantile(0.75))
                    max_val = float(col_data.max())
                    
                    # Additional statistics
                    mean_val = float(col_data.mean())
                    std_val = float(col_data.std()) if len(col_data) > 1 else 0.0
                    var_val = float(col_data.var()) if len(col_data) > 1 else 0.0
                    skew_val = float(col_data.skew()) if len(col_data) > 2 else 0.0
                    kurt_val = float(col_data.kurtosis()) if len(col_data) > 3 else 0.0
                    
                    col_summary.update({
                        "min": min_val,
                        "q25": q25_val,
                        "median": median_val,
                        "q75": q75_val,
                        "max": max_val,
                        "mean": mean_val,
                        "std": std_val,
                        "variance": var_val,
                        "skewness": skew_val,
                        "kurtosis": kurt_val,
                        "range": max_val - min_val,
                        "iqr": q75_val - q25_val,
                        "coefficient_of_variation": (std_val / mean_val * 100) if mean_val != 0 else 0.0
                    })
                else:
                    # No valid data
                    col_summary.update({
                        "min": None,
                        "q25": None,
                        "median": None,
                        "q75": None,
                        "max": None,
                        "mean": None,
                        "std": None,
                        "variance": None,
                        "skewness": None,
                        "kurtosis": None,
                        "range": None,
                        "iqr": None,
                        "coefficient_of_variation": None
                    })
            else:
                value_counts = df[col].value_counts()
                col_summary.update({
                    "most_frequent_value": value_counts.index[0] if len(value_counts) > 0 else None,
                    "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    "most_frequent_percentage": float((value_counts.iloc[0] / len(df)) * 100) if len(value_counts) > 0 else 0
                })
            
            results["column_summary"].append(col_summary)
        
        return serialize_for_json(results)
        
    except Exception as e:
        logger.error(f"Statistical analysis failed for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Statistical analysis failed: {str(e)}"
        )

@router.get("/eda/{dataset_id}/visualizations")
async def generate_visualizations(
    dataset_id: str,
    viz_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate specific visualizations for the dataset
    
    Args:
        dataset_id: ID of the dataset
        viz_types: List of visualization types to generate
                  Options: ['correlation_heatmap', 'distribution_plots', 'missing_values', 
                           'outlier_plots', 'pairplot', 'feature_importance']
    """
    
    # Find dataset file
    dataset_files = list(settings.UPLOAD_DIR.glob(f"{dataset_id}.*"))
    if not dataset_files:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found"
        )
    
    filepath = dataset_files[0]
    
    try:
        # Load dataset
        df = await load_dataset(filepath)
        
        # Initialize visualization generator
        viz_generator = VisualizationGenerator()
        
        visualizations = {}
        
        # Default visualizations if none specified
        if not viz_types:
            viz_types = ['correlation_heatmap', 'distribution_plots', 'missing_values']
        
        for viz_type in viz_types:
            logger.info(f"Generating {viz_type} visualization for dataset {dataset_id}")
            
            if viz_type == 'correlation_heatmap':
                viz = await viz_generator.create_correlation_heatmap(df)
                visualizations[viz_type] = viz
                
            elif viz_type == 'distribution_plots':
                viz = await viz_generator.create_distribution_plots(df)
                visualizations[viz_type] = viz
                
            elif viz_type == 'missing_values':
                viz = await viz_generator.create_missing_values_plot(df)
                visualizations[viz_type] = viz
                
            elif viz_type == 'outlier_plots':
                viz = await viz_generator.create_outlier_plots(df)
                visualizations[viz_type] = viz
                
            elif viz_type == 'pairplot':
                viz = await viz_generator.create_pairplot(df)
                visualizations[viz_type] = viz
                
            elif viz_type == 'feature_importance':
                viz = await viz_generator.create_feature_importance_plot(df)
                visualizations[viz_type] = viz
                
            elif viz_type == 'detailed_univariate':
                viz = await viz_generator.create_detailed_univariate_analysis(df)
                visualizations[viz_type] = viz
                
            elif viz_type == 'bivariate_analysis':
                viz = await viz_generator.create_bivariate_analysis(df)
                visualizations[viz_type] = viz
        
        return serialize_for_json({
            "dataset_id": dataset_id,
            "visualizations": visualizations,
            "generated_at": pd.Timestamp.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Visualization generation failed for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Visualization generation failed: {str(e)}"
        )

async def generate_comprehensive_report(dataset_id: str, df: pd.DataFrame, eda_results: Dict[str, Any]):
    """Background task to generate comprehensive EDA report"""
    
    try:
        logger.info(f"Generating comprehensive EDA report for dataset {dataset_id}")
        
        # Generate HTML report
        from app.ml.report_generator import ReportGenerator
        report_generator = ReportGenerator()
        
        html_report = await report_generator.generate_eda_report(df, eda_results)
        
        # Save report
        report_path = settings.REPORTS_DIR / f"eda_report_{dataset_id}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        # Save report metadata
        report_metadata = {
            "dataset_id": dataset_id,
            "report_type": "comprehensive_eda",
            "report_path": str(report_path),
            "generated_at": pd.Timestamp.now().isoformat(),
            "file_size": report_path.stat().st_size
        }
        
        local_storage.save_json(f"report_{dataset_id}_eda", report_metadata)
        
        logger.info(f"Comprehensive EDA report generated for dataset {dataset_id}")
        
    except Exception as e:
        logger.error(f"Failed to generate comprehensive report for dataset {dataset_id}: {str(e)}")

@router.get("/eda/{dataset_id}/report")
async def get_eda_report(dataset_id: str):
    """Get the comprehensive EDA report for a dataset"""
    
    try:
        # Check if report exists
        report_metadata = local_storage.load_json(f"report_{dataset_id}_eda")
        
        if not report_metadata:
            raise HTTPException(
                status_code=404,
                detail="EDA report not found. Run EDA analysis first."
            )
        
        report_path = Path(report_metadata["report_path"])
        
        if not report_path.exists():
            raise HTTPException(
                status_code=404,
                detail="EDA report file not found"
            )
        
        # Return report metadata and download link
        return {
            "dataset_id": dataset_id,
            "report_metadata": report_metadata,
            "download_url": f"/api/v1/eda/{dataset_id}/report/download"
        }
        
    except Exception as e:
        logger.error(f"Error retrieving EDA report for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve EDA report"
        )

@router.get("/eda/{dataset_id}/ai-overview")
async def get_ai_overview(dataset_id: str) -> Dict[str, Any]:
    """
    Get AI-generated overview of the dataset (2-line summary)
    """
    try:
        # Get dataset info from database or local storage
        dataset_info = await db_manager.get_dataset_metadata(dataset_id)
        if not dataset_info:
            # Fallback to local storage
            dataset_info = local_storage.load_json(f"dataset_{dataset_id}")
            if not dataset_info:
                raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Get file path
        filepath = Path(settings.UPLOAD_DIR) / dataset_info["filename"]
        if not filepath.exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")
        
        # Load dataset
        df = await load_dataset(filepath)
        
        # Generate AI overview using new AI agent
        overview_result = await ai_agent.analyze_dataset_overview(df, dataset_info.get("original_filename", "dataset"))
        
        return {
            "dataset_id": dataset_id,
            "ai_overview": overview_result.get("data", {}),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate AI overview for dataset {dataset_id}: {str(e)}")
        # Return fallback response
        return {
            "dataset_id": dataset_id,
            "ai_overview": {
                "overview": {
                    "overview_line_1": "This dataset contains structured data for analysis.",
                    "overview_line_2": "AI overview generation is temporarily unavailable."
                }
            },
            "timestamp": datetime.utcnow().isoformat(),
            "fallback": True
        }

@router.get("/eda/{dataset_id}/ai-summary")
async def get_ai_summary(dataset_id: str) -> Dict[str, Any]:
    """
    Get AI-generated statistical summary of the dataset (10 key insights)
    """
    try:
        # Get dataset info from database or local storage
        dataset_info = await db_manager.get_dataset_metadata(dataset_id)
        if not dataset_info:
            # Fallback to local storage
            dataset_info = local_storage.load_json(f"dataset_{dataset_id}")
            if not dataset_info:
                raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Get file path
        filepath = Path(settings.UPLOAD_DIR) / dataset_info["filename"]
        if not filepath.exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")
        
        # Load dataset
        df = await load_dataset(filepath)
        
        # Initialize EDA processor for basic stats
        eda_processor = EDAProcessor()
        basic_stats = await eda_processor.generate_basic_statistics(df)
        
        # Generate AI summary using new AI agent
        summary_result = await ai_agent.analyze_statistical_summary(df, basic_stats, dataset_info.get("original_filename", "dataset"))
        
        return {
            "dataset_id": dataset_id,
            "ai_summary": summary_result.get("data", {}),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate AI summary for dataset {dataset_id}: {str(e)}")
        # Return fallback response
        fallback_points = [
            "Dataset contains multiple columns with mixed data types",
            "Missing values detected in some columns requiring attention", 
            "Numerical columns show varying distributions and ranges",
            "Categorical variables have different cardinality levels",
            "Data quality assessment shows areas for improvement",
            "Statistical analysis reveals interesting patterns in the data",
            "Correlation analysis may reveal relationships between variables",
            "Outlier detection could identify anomalous data points",
            "Feature engineering opportunities exist for model improvement",
            "Further domain expertise recommended for deeper insights"
        ]
        
        return {
            "dataset_id": dataset_id,
            "ai_summary": {
                "summary": {
                    "summary_points": fallback_points
                }
            },
            "timestamp": datetime.utcnow().isoformat(),
            "fallback": True
        }
