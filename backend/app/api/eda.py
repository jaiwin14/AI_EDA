"""
EDA API Endpoints
Automated exploratory data analysis endpoints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional
import json

from app.core.config import settings
from app.utils.json_utils import serialize_for_json
from app.core.database import db_manager, local_storage
from app.ml.eda_pipeline import EDAProcessor
from app.ml.visualization_generator import VisualizationGenerator
from app.api.upload import load_dataset

logger = logging.getLogger(__name__)
router = APIRouter()

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
        # Get results from database
        results = await db_manager.get_eda_results(dataset_id, analysis_type)
        
        if not results:
            # If no results found, return empty structure instead of 404
            logger.warning(f"No EDA results found for dataset {dataset_id}")
            return {
                "dataset_id": dataset_id,
                "analyses": {},
                "summary": {
                    "total_analyses": 0,
                    "completed_at": None,
                    "status": "no_results"
                }
            }
        
        return {
            "dataset_id": dataset_id,
            "results": results
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
        
        # Generate comprehensive descriptive statistics for numeric columns
        numeric_describe = {}
        if len(numeric_cols) > 0:
            describe_df = df[numeric_cols].describe()
            for col in numeric_cols:
                numeric_describe[col] = {
                    "count": float(describe_df.loc['count', col]),
                    "mean": float(describe_df.loc['mean', col]),
                    "std": float(describe_df.loc['std', col]),
                    "min": float(describe_df.loc['min', col]),
                    "25%": float(describe_df.loc['25%', col]),
                    "50%": float(describe_df.loc['50%', col]),
                    "75%": float(describe_df.loc['75%', col]),
                    "max": float(describe_df.loc['max', col])
                }

        # Generate categorical summary
        categorical_describe = {}
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            categorical_describe[col] = {
                "count": int(df[col].count()),
                "unique": int(df[col].nunique()),
                "top": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "freq": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "unique_ratio": float(df[col].nunique() / len(df)),
                "value_counts": dict(value_counts.head(10).to_dict())
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
                    "describe": numeric_describe
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
