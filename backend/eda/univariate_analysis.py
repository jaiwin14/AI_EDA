"""Univariate Analysis Module with Interactive Visualizations"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.ai_utils import generate_insight
from utils.serialization_fixed import to_json_serializable

def run(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform univariate analysis for each column with appropriate visualizations
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing univariate analysis results and visualizations
    """
    try:
        results = {
            "text": f"Univariate analysis completed for {len(df.columns)} columns",
            "column_analyses": {},
            "summary": {},
            "ai_insights": ""
        }
        
        # Analyze each column
        for column in df.columns:
            col_analysis = analyze_column(df[column], column)
            results["column_analyses"][column] = col_analysis
        
        # Generate summary statistics
        results["summary"] = generate_summary_stats(df, results["column_analyses"])
        
        # Generate AI insights
        insight_prompt = f"""
        Analyze the univariate analysis results for this dataset:
        - Total columns: {len(df.columns)}
        - Column types: {results['summary']['column_types']}
        
        For each column type, provide insights on:
        1. Distribution patterns and characteristics
        2. Potential outliers or anomalies
        3. Data quality issues
        4. Recommendations for further analysis
        5. Key observations about the data structure
        """
        
        try:
            results["ai_insights"] = generate_insight(insight_prompt, context=results["column_analyses"])
        except Exception as e:
            results["ai_insights"] = f"Unable to generate AI insights: {str(e)}"
        
        return results
        
    except Exception as e:
        return {
            "text": f"Error in univariate analysis: {str(e)}",
            "error": str(e)
        }

def analyze_column(series: pd.Series, column_name: str) -> Dict[str, Any]:
    """Analyze a single column and create appropriate visualizations"""
    
    analysis = {
        "column_name": column_name,
        "data_type": str(series.dtype),
        "missing_count": int(series.isnull().sum()),
        "missing_percentage": round((series.isnull().sum() / len(series)) * 100, 2),
        "unique_count": int(series.nunique()),
        "unique_percentage": round((series.nunique() / len(series)) * 100, 2)
    }
    
    # Determine column type for appropriate analysis
    if pd.api.types.is_numeric_dtype(series):
        analysis.update(analyze_numeric_column(series))
    elif pd.api.types.is_datetime64_dtype(series):
        analysis.update(analyze_datetime_column(series))
    else:
        analysis.update(analyze_categorical_column(series))
    
    return analysis

def analyze_numeric_column(series: pd.Series) -> Dict[str, Any]:
    """Analyze numeric column with statistical measures and visualizations"""
    
    # Remove nulls for analysis
    clean_series = series.dropna()
    
    if len(clean_series) == 0:
        return {"error": "No valid numeric data"}
    
    # Statistical measures
    stats = {
        "count": int(len(clean_series)),
        "mean": float(clean_series.mean()),
        "median": float(clean_series.median()),
        "std": float(clean_series.std()),
        "min": float(clean_series.min()),
        "max": float(clean_series.max()),
        "q1": float(clean_series.quantile(0.25)),
        "q3": float(clean_series.quantile(0.75)),
        "iqr": float(clean_series.quantile(0.75) - clean_series.quantile(0.25)),
        "skewness": float(clean_series.skew()),
        "kurtosis": float(clean_series.kurtosis())
    }
    
    # Outlier detection using IQR method
    q1, q3 = clean_series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
    stats["outlier_count"] = int(len(outliers))
    stats["outlier_percentage"] = round((len(outliers) / len(clean_series)) * 100, 2)
    
    # Create visualizations
    visualizations = {}
    
    # Histogram
    hist_fig = go.Figure(data=go.Histogram(
        x=clean_series,
        nbinsx=min(50, len(clean_series.unique())),
        name="Histogram"
    ))
    hist_fig.update_layout(
        title=f"Distribution of {series.name}",
        xaxis_title=series.name,
        yaxis_title="Frequency",
        height=400
    )
    visualizations["histogram"] = hist_fig.to_dict()
    
    # Box plot
    box_fig = go.Figure(data=go.Box(
        y=clean_series,
        name="Box Plot",
        boxpoints="outliers"
    ))
    box_fig.update_layout(
        title=f"Box Plot of {series.name}",
        yaxis_title=series.name,
        height=400
    )
    visualizations["boxplot"] = box_fig.to_dict()
    
    # Q-Q plot for normality check
    try:
        from scipy import stats as scipy_stats
        qq_data = scipy_stats.probplot(clean_series, dist="norm")
        
        qq_fig = go.Figure()
        qq_fig.add_trace(go.Scatter(
            x=qq_data[0][0],
            y=qq_data[0][1],
            mode='markers',
            name='Data Points'
        ))
        qq_fig.add_trace(go.Scatter(
            x=qq_data[0][0],
            y=qq_data[1][0] * qq_data[0][0] + qq_data[1][1],
            mode='lines',
            name='Theoretical Line'
        ))
        qq_fig.update_layout(
            title=f"Q-Q Plot of {series.name}",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            height=400
        )
        visualizations["qqplot"] = qq_fig.to_dict()
    except ImportError:
        visualizations["qqplot"] = {"error": "scipy not available for Q-Q plot"}
    
    return {
        "type": "numeric",
        "statistics": stats,
        "visualizations": visualizations
    }

def analyze_categorical_column(series: pd.Series) -> Dict[str, Any]:
    """Analyze categorical column with frequency analysis and visualizations"""
    
    # Remove nulls for analysis
    clean_series = series.dropna()
    
    if len(clean_series) == 0:
        return {"error": "No valid categorical data"}
    
    # Frequency analysis
    value_counts = clean_series.value_counts()
    value_counts_percent = clean_series.value_counts(normalize=True) * 100
    
    stats = {
        "count": int(len(clean_series)),
        "unique_values": int(len(value_counts)),
        "most_common": value_counts.index[0] if len(value_counts) > 0 else None,
        "most_common_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
        "most_common_percentage": round(float(value_counts_percent.iloc[0]), 2) if len(value_counts_percent) > 0 else 0,
        "least_common": value_counts.index[-1] if len(value_counts) > 0 else None,
        "least_common_count": int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
        "least_common_percentage": round(float(value_counts_percent.iloc[-1]), 2) if len(value_counts_percent) > 0 else 0
    }
    
    # Create visualizations
    visualizations = {}
    
    # Bar chart (top 20 categories)
    top_values = value_counts.head(20)
    bar_fig = go.Figure(data=go.Bar(
        x=top_values.index.astype(str),
        y=top_values.values,
        name="Frequency"
    ))
    bar_fig.update_layout(
        title=f"Top Categories in {series.name}",
        xaxis_title=series.name,
        yaxis_title="Count",
        height=400,
        xaxis_tickangle=-45
    )
    visualizations["barchart"] = bar_fig.to_dict()
    
    # Pie chart (top 10 categories, rest as "Others")
    if len(value_counts) > 10:
        top_10 = value_counts.head(10)
        others_count = value_counts.iloc[10:].sum()
        pie_data = pd.concat([top_10, pd.Series([others_count], index=['Others'])])
    else:
        pie_data = value_counts
    
    pie_fig = go.Figure(data=go.Pie(
        labels=pie_data.index.astype(str),
        values=pie_data.values,
        hole=0.3
    ))
    pie_fig.update_layout(
        title=f"Distribution of {series.name}",
        height=400
    )
    visualizations["piechart"] = pie_fig.to_dict()
    
    return {
        "type": "categorical",
        "statistics": stats,
        "value_counts": value_counts.to_dict(),
        "value_counts_percent": value_counts_percent.to_dict(),
        "visualizations": visualizations
    }

def analyze_datetime_column(series: pd.Series) -> Dict[str, Any]:
    """Analyze datetime column with temporal patterns"""
    
    # Remove nulls for analysis
    clean_series = series.dropna()
    
    if len(clean_series) == 0:
        return {"error": "No valid datetime data"}
    
    # Temporal statistics
    stats = {
        "count": int(len(clean_series)),
        "min_date": clean_series.min().isoformat(),
        "max_date": clean_series.max().isoformat(),
        "date_range_days": int((clean_series.max() - clean_series.min()).days),
        "unique_dates": int(clean_series.dt.date.nunique())
    }
    
    # Create visualizations
    visualizations = {}
    
    # Time series plot
    time_series = clean_series.value_counts().sort_index()
    ts_fig = go.Figure(data=go.Scatter(
        x=time_series.index,
        y=time_series.values,
        mode='lines+markers',
        name="Time Series"
    ))
    ts_fig.update_layout(
        title=f"Time Series of {series.name}",
        xaxis_title="Date",
        yaxis_title="Frequency",
        height=400
    )
    visualizations["timeseries"] = ts_fig.to_dict()
    
    # Monthly/Yearly distribution
    try:
        monthly_dist = clean_series.dt.month.value_counts().sort_index()
        monthly_fig = go.Figure(data=go.Bar(
            x=monthly_dist.index,
            y=monthly_dist.values,
            name="Monthly Distribution"
        ))
        monthly_fig.update_layout(
            title=f"Monthly Distribution of {series.name}",
            xaxis_title="Month",
            yaxis_title="Count",
            height=400
        )
        visualizations["monthly_dist"] = monthly_fig.to_dict()
    except:
        visualizations["monthly_dist"] = {"error": "Unable to create monthly distribution"}
    
    return {
        "type": "datetime",
        "statistics": stats,
        "visualizations": visualizations
    }

def generate_summary_stats(df: pd.DataFrame, column_analyses: Dict) -> Dict[str, Any]:
    """Generate summary statistics across all columns"""
    
    summary = {
        "total_columns": len(df.columns),
        "column_types": {},
        "missing_data_summary": {},
        "outlier_summary": {}
    }
    
    # Count column types
    type_counts = {}
    for col_analysis in column_analyses.values():
        if "type" in col_analysis:
            col_type = col_analysis["type"]
            type_counts[col_type] = type_counts.get(col_type, 0) + 1
    
    summary["column_types"] = type_counts
    
    # Missing data summary
    total_missing = sum(col_analysis.get("missing_count", 0) for col_analysis in column_analyses.values())
    summary["missing_data_summary"] = {
        "total_missing": total_missing,
        "total_cells": len(df) * len(df.columns),
        "missing_percentage": round((total_missing / (len(df) * len(df.columns))) * 100, 2)
    }
    
    # Outlier summary for numeric columns
    outlier_count = 0
    for col_analysis in column_analyses.values():
        if col_analysis.get("type") == "numeric" and "statistics" in col_analysis:
            outlier_count += col_analysis["statistics"].get("outlier_count", 0)
    
    summary["outlier_summary"] = {
        "total_outliers": outlier_count
    }
    
    return summary
