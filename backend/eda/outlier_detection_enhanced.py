"""Enhanced Outlier Detection and Treatment Module with AI Insights"""
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
    Enhanced outlier detection using multiple methods (IQR, Z-score) with AI insights
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing outlier analysis results and visualizations
    """
    try:
        # Select numeric columns only
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            return {
                "text": "No numeric columns found for outlier detection",
                "error": "No numeric columns available"
            }
        
        results = {
            "text": f"Enhanced outlier detection completed for {len(numeric_columns)} numeric columns",
            "outlier_analysis": {},
            "summary": {},
            "ai_insights": "",
            "treatment_recommendations": {}
        }
        
        # Analyze each numeric column
        for column in numeric_columns:
            col_analysis = analyze_column_outliers(df[column], column)
            results["outlier_analysis"][column] = col_analysis
        
        # Generate summary statistics
        results["summary"] = generate_outlier_summary(results["outlier_analysis"])
        
        # Generate AI insights
        insight_prompt = f"""
        Analyze the outlier detection results for this dataset:
        - Total numeric columns: {len(numeric_columns)}
        - Total outliers detected: {results['summary']['total_outliers']}
        - Outlier percentage: {results['summary']['total_outlier_percentage']:.2f}%
        
        For each detection method (IQR and Z-score), provide insights on:
        1. Which method is more appropriate for this dataset and why
        2. Potential causes of outliers (data entry errors, natural variation, etc.)
        3. Impact of outliers on statistical analysis
        4. Recommendations for outlier treatment (removal, capping, transformation)
        5. Business context considerations for outlier handling
        
        Provide specific recommendations for each column with significant outliers.
        """
        
        try:
            results["ai_insights"] = generate_insight(insight_prompt, context=results["outlier_analysis"])
        except Exception as e:
            results["ai_insights"] = f"Unable to generate AI insights: {str(e)}"
        
        # Generate treatment recommendations
        results["treatment_recommendations"] = generate_treatment_recommendations(results["outlier_analysis"])
        
        return results
        
    except Exception as e:
        return {
            "text": f"Error in outlier detection: {str(e)}",
            "error": str(e)
        }

def analyze_column_outliers(series: pd.Series, column_name: str) -> Dict[str, Any]:
    """Analyze outliers in a single column using multiple methods"""
    
    # Remove nulls for analysis
    clean_series = series.dropna()
    
    if len(clean_series) == 0:
        return {"error": "No valid numeric data"}
    
    analysis = {
        "column_name": column_name,
        "total_values": int(len(clean_series)),
        "detection_methods": {}
    }
    
    # IQR Method
    iqr_results = detect_outliers_iqr(clean_series)
    analysis["detection_methods"]["iqr"] = iqr_results
    
    # Z-Score Method
    zscore_results = detect_outliers_zscore(clean_series)
    analysis["detection_methods"]["zscore"] = zscore_results
    
    # Combined analysis
    analysis["combined"] = combine_outlier_methods(iqr_results, zscore_results)
    
    # Create visualizations
    analysis["visualizations"] = create_outlier_visualizations(clean_series, iqr_results, zscore_results)
    
    return analysis

def detect_outliers_iqr(series: pd.Series) -> Dict[str, Any]:
    """Detect outliers using IQR method"""
    
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    outlier_indices = outliers.index.tolist()
    
    return {
        "method": "IQR",
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outlier_count": int(len(outliers)),
        "outlier_percentage": round((len(outliers) / len(series)) * 100, 2),
        "outlier_indices": outlier_indices,
        "outlier_values": outliers.tolist(),
        "min_outlier": float(outliers.min()) if not outliers.empty else None,
        "max_outlier": float(outliers.max()) if not outliers.empty else None
    }

def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> Dict[str, Any]:
    """Detect outliers using Z-score method"""
    
    z_scores = np.abs((series - series.mean()) / series.std())
    outliers = series[z_scores > threshold]
    outlier_indices = outliers.index.tolist()
    
    return {
        "method": "Z-Score",
        "threshold": threshold,
        "mean": float(series.mean()),
        "std": float(series.std()),
        "outlier_count": int(len(outliers)),
        "outlier_percentage": round((len(outliers) / len(series)) * 100, 2),
        "outlier_indices": outlier_indices,
        "outlier_values": outliers.tolist(),
        "min_outlier": float(outliers.min()) if not outliers.empty else None,
        "max_outlier": float(outliers.max()) if not outliers.empty else None,
        "max_z_score": float(z_scores.max()) if not z_scores.empty else 0
    }

def combine_outlier_methods(iqr_results: Dict, zscore_results: Dict) -> Dict[str, Any]:
    """Combine results from both outlier detection methods"""
    
    iqr_outliers = set(iqr_results["outlier_indices"])
    zscore_outliers = set(zscore_results["outlier_indices"])
    
    # Find common outliers
    common_outliers = iqr_outliers.intersection(zscore_outliers)
    
    # Find outliers detected by only one method
    only_iqr = iqr_outliers - zscore_outliers
    only_zscore = zscore_outliers - iqr_outliers
    
    return {
        "common_outliers": list(common_outliers),
        "only_iqr": list(only_iqr),
        "only_zscore": list(only_zscore),
        "total_unique_outliers": len(iqr_outliers.union(zscore_outliers)),
        "agreement_percentage": round((len(common_outliers) / max(len(iqr_outliers), len(zscore_outliers))) * 100, 2) if max(len(iqr_outliers), len(zscore_outliers)) > 0 else 0
    }

def create_outlier_visualizations(series: pd.Series, iqr_results: Dict, zscore_results: Dict) -> Dict[str, Any]:
    """Create visualizations for outlier analysis"""
    
    visualizations = {}
    
    # Box plot with outliers
    box_fig = go.Figure(data=go.Box(
        y=series,
        name="Box Plot",
        boxpoints="outliers",
        jitter=0.3,
        pointpos=-1.8
    ))
    box_fig.update_layout(
        title=f"Box Plot with Outliers - {series.name}",
        yaxis_title=series.name,
        height=400
    )
    visualizations["boxplot"] = box_fig.to_dict()
    
    # Histogram with outlier regions highlighted
    hist_fig = go.Figure()
    
    # Main histogram
    hist_fig.add_trace(go.Histogram(
        x=series,
        nbinsx=50,
        name="Distribution",
        marker_color="lightblue"
    ))
    
    # Add outlier regions
    if iqr_results["outlier_count"] > 0:
        # Lower outliers
        lower_outliers = series[series < iqr_results["lower_bound"]]
        if len(lower_outliers) > 0:
            hist_fig.add_trace(go.Histogram(
                x=lower_outliers,
                nbinsx=10,
                name="Lower Outliers (IQR)",
                marker_color="red"
            ))
        
        # Upper outliers
        upper_outliers = series[series > iqr_results["upper_bound"]]
        if len(upper_outliers) > 0:
            hist_fig.add_trace(go.Histogram(
                x=upper_outliers,
                nbinsx=10,
                name="Upper Outliers (IQR)",
                marker_color="orange"
            ))
    
    hist_fig.update_layout(
        title=f"Distribution with Outlier Regions - {series.name}",
        xaxis_title=series.name,
        yaxis_title="Frequency",
        height=400,
        barmode="overlay"
    )
    visualizations["histogram"] = hist_fig.to_dict()
    
    # Z-score plot
    z_scores = np.abs((series - series.mean()) / series.std())
    zscore_fig = go.Figure(data=go.Scatter(
        x=series,
        y=z_scores,
        mode='markers',
        marker=dict(
            size=8,
            color=z_scores,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Z-Score")
        ),
        name="Z-Scores"
    ))
    
    # Add threshold line
    zscore_fig.add_hline(
        y=zscore_results["threshold"],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold ({zscore_results['threshold']})"
    )
    
    zscore_fig.update_layout(
        title=f"Z-Score Analysis - {series.name}",
        xaxis_title=series.name,
        yaxis_title="Absolute Z-Score",
        height=400
    )
    visualizations["zscore_plot"] = zscore_fig.to_dict()
    
    return visualizations

def generate_outlier_summary(outlier_analysis: Dict) -> Dict[str, Any]:
    """Generate summary statistics for outlier analysis"""
    
    summary = {
        "total_columns": len(outlier_analysis),
        "total_outliers": 0,
        "columns_with_outliers": 0,
        "outlier_methods_comparison": {
            "iqr_total": 0,
            "zscore_total": 0,
            "common_total": 0
        }
    }
    
    for column, analysis in outlier_analysis.items():
        if "error" in analysis:
            continue
            
        iqr_count = analysis["detection_methods"]["iqr"]["outlier_count"]
        zscore_count = analysis["detection_methods"]["zscore"]["outlier_count"]
        common_count = len(analysis["combined"]["common_outliers"])
        
        summary["outlier_methods_comparison"]["iqr_total"] += iqr_count
        summary["outlier_methods_comparison"]["zscore_total"] += zscore_count
        summary["outlier_methods_comparison"]["common_total"] += common_count
        
        if iqr_count > 0 or zscore_count > 0:
            summary["columns_with_outliers"] += 1
    
    summary["total_outliers"] = max(
        summary["outlier_methods_comparison"]["iqr_total"],
        summary["outlier_methods_comparison"]["zscore_total"]
    )
    
    if summary["total_columns"] > 0:
        summary["total_outlier_percentage"] = round(
            (summary["total_outliers"] / (summary["total_columns"] * 1000)) * 100, 2  # Assuming average 1000 values per column
        )
    
    return summary

def generate_treatment_recommendations(outlier_analysis: Dict) -> Dict[str, Any]:
    """Generate recommendations for outlier treatment"""
    
    recommendations = {}
    
    for column, analysis in outlier_analysis.items():
        if "error" in analysis:
            continue
            
        iqr_count = analysis["detection_methods"]["iqr"]["outlier_count"]
        zscore_count = analysis["detection_methods"]["zscore"]["outlier_count"]
        iqr_percentage = analysis["detection_methods"]["iqr"]["outlier_percentage"]
        
        recommendation = {
            "outlier_count": max(iqr_count, zscore_count),
            "outlier_percentage": iqr_percentage,
            "recommended_action": "none",
            "reasoning": "",
            "treatment_method": None
        }
        
        if iqr_percentage < 1:
            recommendation["recommended_action"] = "ignore"
            recommendation["reasoning"] = "Very few outliers (<1%), likely natural variation"
        elif iqr_percentage < 5:
            recommendation["recommended_action"] = "investigate"
            recommendation["reasoning"] = "Moderate number of outliers, investigate for data quality issues"
            recommendation["treatment_method"] = "capping"
        else:
            recommendation["recommended_action"] = "treat"
            recommendation["reasoning"] = "High number of outliers, consider treatment"
            recommendation["treatment_method"] = "removal_or_capping"
        
        recommendations[column] = recommendation
    
    return recommendations

def treat_outliers(df: pd.DataFrame, treatment_method: str = "capping", columns: List[str] = None) -> Dict[str, Any]:
    """
    Treat outliers in the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    treatment_method : str
        Method to treat outliers: "capping", "removal", "transformation"
    columns : List[str]
        Specific columns to treat (if None, treat all numeric columns)
    
    Returns:
    --------
    Dict[str, Any]
        Results of outlier treatment
    """
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    columns_to_treat = columns or numeric_columns
    
    df_treated = df.copy()
    treatment_results = {}
    
    for column in columns_to_treat:
        if column not in numeric_columns:
            continue
            
        series = df[column].dropna()
        if len(series) == 0:
            continue
        
        if treatment_method == "capping":
            # IQR capping
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Cap outliers
            df_treated[column] = df_treated[column].clip(lower=lower_bound, upper=upper_bound)
            
            treatment_results[column] = {
                "method": "capping",
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "original_outliers": int(len(series[(series < lower_bound) | (series > upper_bound)])),
                "values_capped": int(len(df[column][(df[column] < lower_bound) | (df[column] > upper_bound)]))
            }
            
        elif treatment_method == "removal":
            # Remove rows with outliers
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
            df_treated = df_treated[~outlier_mask]
            
            treatment_results[column] = {
                "method": "removal",
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "rows_removed": int(outlier_mask.sum()),
                "remaining_rows": int(len(df_treated))
            }
            
        elif treatment_method == "transformation":
            # Log transformation for positive skewed data
            if series.min() > 0:
                df_treated[column] = np.log1p(df_treated[column])
                treatment_results[column] = {
                    "method": "log_transformation",
                    "transformation": "log1p"
                }
            else:
                # Square root transformation
                df_treated[column] = np.sqrt(np.abs(df_treated[column]))
                treatment_results[column] = {
                    "method": "sqrt_transformation",
                    "transformation": "sqrt"
                }
    
    return {
        "treated_dataframe": df_treated,
        "treatment_results": treatment_results,
        "original_shape": df.shape,
        "treated_shape": df_treated.shape,
        "rows_removed": df.shape[0] - df_treated.shape[0]
    }
