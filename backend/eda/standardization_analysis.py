"""Standardization Analysis Module with AI Recommendations"""
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
    Analyze the need for standardization and recommend appropriate methods
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing standardization analysis and recommendations
    """
    try:
        # Select numeric columns only
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            return {
                "text": "No numeric columns found for standardization analysis",
                "error": "No numeric columns available"
            }
        
        results = {
            "text": f"Standardization analysis completed for {len(numeric_columns)} numeric columns",
            "column_analysis": {},
            "overall_recommendation": {},
            "ai_insights": "",
            "visualizations": {}
        }
        
        # Analyze each numeric column
        for column in numeric_columns:
            col_analysis = analyze_column_standardization(df[column], column)
            results["column_analysis"][column] = col_analysis
        
        # Generate overall recommendation
        results["overall_recommendation"] = generate_overall_recommendation(results["column_analysis"])
        
        # Generate AI insights
        insight_prompt = f"""
        Analyze the standardization analysis results for this dataset:
        - Total numeric columns: {len(numeric_columns)}
        - Columns needing standardization: {sum(1 for col in results['column_analysis'].values() if col.get('needs_standardization', False))}
        
        For each column that needs standardization, provide insights on:
        1. Why standardization is recommended (scale differences, algorithm requirements, etc.)
        2. Which method is most appropriate (StandardScaler vs MinMaxScaler) and why
        3. Potential impact on machine learning models
        4. Business context considerations
        5. Implementation recommendations
        
        Explain the differences between StandardScaler and MinMaxScaler:
        - StandardScaler: Standardizes features by removing the mean and scaling to unit variance
        - MinMaxScaler: Scales features to a given range (default [0,1])
        
        Consider factors like:
        - Presence of outliers
        - Data distribution (normal vs skewed)
        - Algorithm requirements (some algorithms are sensitive to feature scales)
        - Business interpretability needs
        """
        
        try:
            results["ai_insights"] = generate_insight(insight_prompt, context=results["column_analysis"])
        except Exception as e:
            results["ai_insights"] = f"Unable to generate AI insights: {str(e)}"
        
        # Create visualizations
        results["visualizations"] = create_standardization_visualizations(df, results["column_analysis"])
        
        return results
        
    except Exception as e:
        return {
            "text": f"Error in standardization analysis: {str(e)}",
            "error": str(e)
        }

def analyze_column_standardization(series: pd.Series, column_name: str) -> Dict[str, Any]:
    """Analyze standardization needs for a single column"""
    
    # Remove nulls for analysis
    clean_series = series.dropna()
    
    if len(clean_series) == 0:
        return {"error": "No valid numeric data"}
    
    # Basic statistics
    mean_val = float(clean_series.mean())
    std_val = float(clean_series.std())
    min_val = float(clean_series.min())
    max_val = float(clean_series.max())
    range_val = max_val - min_val
    
    # Calculate coefficient of variation (CV)
    cv = (std_val / abs(mean_val)) * 100 if mean_val != 0 else float('inf')
    
    # Check for outliers using IQR method
    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
    outlier_percentage = (len(outliers) / len(clean_series)) * 100
    
    # Calculate skewness and kurtosis
    skewness = float(clean_series.skew())
    kurtosis = float(clean_series.kurtosis())
    
    # Determine if standardization is needed
    needs_standardization = determine_standardization_need(
        mean_val, std_val, range_val, cv, outlier_percentage, skewness
    )
    
    # Recommend appropriate method
    recommended_method = recommend_standardization_method(
        mean_val, std_val, range_val, cv, outlier_percentage, skewness, kurtosis
    )
    
    analysis = {
        "column_name": column_name,
        "statistics": {
            "count": int(len(clean_series)),
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "range": range_val,
            "coefficient_of_variation": round(cv, 2),
            "skewness": round(skewness, 3),
            "kurtosis": round(kurtosis, 3)
        },
        "outlier_analysis": {
            "outlier_count": int(len(outliers)),
            "outlier_percentage": round(outlier_percentage, 2),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound)
        },
        "standardization_assessment": {
            "needs_standardization": needs_standardization,
            "recommended_method": recommended_method,
            "reasoning": generate_standardization_reasoning(
                mean_val, std_val, range_val, cv, outlier_percentage, skewness, recommended_method
            )
        }
    }
    
    return analysis

def determine_standardization_need(mean: float, std: float, range_val: float, cv: float, 
                                 outlier_percentage: float, skewness: float) -> bool:
    """Determine if standardization is needed based on various factors"""
    
    # Multiple criteria for determining standardization need
    criteria = []
    
    # 1. Large scale differences (coefficient of variation)
    if cv > 50:  # High variability relative to mean
        criteria.append("high_variability")
    
    # 2. Large range
    if range_val > 1000:  # Arbitrary threshold for large range
        criteria.append("large_range")
    
    # 3. High outlier percentage
    if outlier_percentage > 5:
        criteria.append("high_outliers")
    
    # 4. High skewness (non-normal distribution)
    if abs(skewness) > 1:
        criteria.append("skewed_distribution")
    
    # 5. Mean not close to zero (for algorithms sensitive to feature scales)
    if abs(mean) > std:
        criteria.append("non_zero_mean")
    
    # Standardization is recommended if at least 2 criteria are met
    return len(criteria) >= 2

def recommend_standardization_method(mean: float, std: float, range_val: float, cv: float,
                                   outlier_percentage: float, skewness: float, kurtosis: float) -> str:
    """Recommend appropriate standardization method"""
    
    # Factors favoring StandardScaler
    standard_scaler_factors = 0
    
    # 1. Normal distribution (low skewness)
    if abs(skewness) < 1:
        standard_scaler_factors += 1
    
    # 2. Low outlier percentage
    if outlier_percentage < 3:
        standard_scaler_factors += 1
    
    # 3. Reasonable coefficient of variation
    if 10 < cv < 200:
        standard_scaler_factors += 1
    
    # 4. Mean close to zero
    if abs(mean) < std:
        standard_scaler_factors += 1
    
    # Factors favoring MinMaxScaler
    minmax_scaler_factors = 0
    
    # 1. High outlier percentage
    if outlier_percentage > 5:
        minmax_scaler_factors += 1
    
    # 2. Skewed distribution
    if abs(skewness) > 1:
        minmax_scaler_factors += 1
    
    # 3. Need for bounded output
    minmax_scaler_factors += 1  # Always a consideration
    
    # 4. High kurtosis (heavy tails)
    if abs(kurtosis) > 3:
        minmax_scaler_factors += 1
    
    # Decision logic
    if standard_scaler_factors > minmax_scaler_factors:
        return "StandardScaler"
    elif minmax_scaler_factors > standard_scaler_factors:
        return "MinMaxScaler"
    else:
        # Tie-breaker: prefer StandardScaler for normal distributions
        return "StandardScaler" if abs(skewness) < 1 else "MinMaxScaler"

def generate_standardization_reasoning(mean: float, std: float, range_val: float, cv: float,
                                     outlier_percentage: float, skewness: float, method: str) -> str:
    """Generate reasoning for standardization recommendation"""
    
    reasons = []
    
    if method == "StandardScaler":
        reasons.append("StandardScaler is recommended because:")
        if abs(skewness) < 1:
            reasons.append("- Data distribution is relatively normal")
        if outlier_percentage < 3:
            reasons.append("- Low outlier presence")
        if abs(mean) < std:
            reasons.append("- Mean is close to zero")
        reasons.append("- Removes mean and scales to unit variance")
        reasons.append("- Good for algorithms that assume normal distribution")
        
    else:  # MinMaxScaler
        reasons.append("MinMaxScaler is recommended because:")
        if outlier_percentage > 5:
            reasons.append("- High outlier presence")
        if abs(skewness) > 1:
            reasons.append("- Skewed distribution")
        reasons.append("- Scales data to bounded range [0,1]")
        reasons.append("- Robust to outliers")
        reasons.append("- Preserves zero entries in sparse data")
    
    return " ".join(reasons)

def generate_overall_recommendation(column_analysis: Dict) -> Dict[str, Any]:
    """Generate overall standardization recommendation"""
    
    total_columns = len(column_analysis)
    columns_needing_standardization = sum(
        1 for col in column_analysis.values() 
        if col.get('standardization_assessment', {}).get('needs_standardization', False)
    )
    
    method_counts = {}
    for col in column_analysis.values():
        if 'standardization_assessment' in col:
            method = col['standardization_assessment'].get('recommended_method', 'None')
            method_counts[method] = method_counts.get(method, 0) + 1
    
    overall_recommendation = {
        "total_columns": total_columns,
        "columns_needing_standardization": columns_needing_standardization,
        "standardization_percentage": round((columns_needing_standardization / total_columns) * 100, 2) if total_columns > 0 else 0,
        "method_distribution": method_counts,
        "overall_need": columns_needing_standardization > total_columns * 0.3  # If more than 30% need standardization
    }
    
    return overall_recommendation

def create_standardization_visualizations(df: pd.DataFrame, column_analysis: Dict) -> Dict[str, Any]:
    """Create visualizations for standardization analysis"""
    
    visualizations = {}
    
    # Select numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) == 0:
        return {"error": "No numeric columns for visualization"}
    
    # 1. Scale comparison chart
    scales_data = []
    for col in numeric_columns:
        if col in column_analysis and 'statistics' in column_analysis[col]:
            stats = column_analysis[col]['statistics']
            scales_data.append({
                'column': col,
                'mean': stats['mean'],
                'std': stats['std'],
                'range': stats['range']
            })
    
    if scales_data:
        # Mean vs Std scatter plot
        mean_std_fig = go.Figure(data=go.Scatter(
            x=[d['mean'] for d in scales_data],
            y=[d['std'] for d in scales_data],
            mode='markers+text',
            text=[d['column'] for d in scales_data],
            textposition="top center",
            marker=dict(size=10, color='blue')
        ))
        mean_std_fig.update_layout(
            title="Mean vs Standard Deviation",
            xaxis_title="Mean",
            yaxis_title="Standard Deviation",
            height=400
        )
        visualizations["mean_std_scatter"] = mean_std_fig.to_dict()
        
        # Range comparison
        range_fig = go.Figure(data=go.Bar(
            x=[d['column'] for d in scales_data],
            y=[d['range'] for d in scales_data],
            name="Range"
        ))
        range_fig.update_layout(
            title="Range Comparison Across Columns",
            xaxis_title="Columns",
            yaxis_title="Range",
            height=400
        )
        visualizations["range_comparison"] = range_fig.to_dict()
    
    # 2. Standardization need summary
    needs_standardization = []
    methods = []
    
    for col in numeric_columns:
        if col in column_analysis and 'standardization_assessment' in column_analysis[col]:
            assessment = column_analysis[col]['standardization_assessment']
            needs_standardization.append(assessment.get('needs_standardization', False))
            methods.append(assessment.get('recommended_method', 'None'))
    
    if needs_standardization:
        # Pie chart of standardization needs
        needs_count = sum(needs_standardization)
        not_needs_count = len(needs_standardization) - needs_count
        
        needs_pie_fig = go.Figure(data=go.Pie(
            labels=['Needs Standardization', 'No Standardization Needed'],
            values=[needs_count, not_needs_count],
            hole=0.3
        ))
        needs_pie_fig.update_layout(
            title="Standardization Need Summary",
            height=400
        )
        visualizations["standardization_needs"] = needs_pie_fig.to_dict()
        
        # Method distribution
        method_counts = {}
        for method in methods:
            method_counts[method] = method_counts.get(method, 0) + 1
        
        method_fig = go.Figure(data=go.Bar(
            x=list(method_counts.keys()),
            y=list(method_counts.values()),
            name="Recommended Method"
        ))
        method_fig.update_layout(
            title="Recommended Standardization Methods",
            xaxis_title="Method",
            yaxis_title="Count",
            height=400
        )
        visualizations["method_distribution"] = method_fig.to_dict()
    
    return visualizations

def apply_standardization(df: pd.DataFrame, method: str = "StandardScaler", columns: List[str] = None) -> Dict[str, Any]:
    """
    Apply standardization to the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    method : str
        Standardization method: "StandardScaler" or "MinMaxScaler"
    columns : List[str]
        Specific columns to standardize (if None, standardize all numeric columns)
    
    Returns:
    --------
    Dict[str, Any]
        Results of standardization
    """
    
    try:
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        columns_to_standardize = columns or numeric_columns
        
        df_standardized = df.copy()
        scaler = None
        
        if method == "StandardScaler":
            scaler = StandardScaler()
        elif method == "MinMaxScaler":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown standardization method: {method}")
        
        # Fit and transform the selected columns
        df_standardized[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
        
        # Get scaling parameters
        scaling_params = {}
        if method == "StandardScaler":
            scaling_params = {
                "mean_": scaler.mean_.tolist(),
                "scale_": scaler.scale_.tolist()
            }
        elif method == "MinMaxScaler":
            scaling_params = {
                "min_": scaler.min_.tolist(),
                "scale_": scaler.scale_.tolist(),
                "data_min_": scaler.data_min_.tolist(),
                "data_max_": scaler.data_max_.tolist()
            }
        
        return {
            "standardized_dataframe": df_standardized,
            "method": method,
            "columns_standardized": columns_to_standardize,
            "scaling_parameters": scaling_params,
            "original_shape": df.shape,
            "standardized_shape": df_standardized.shape
        }
        
    except ImportError:
        return {
            "error": "scikit-learn not available for standardization"
        }
    except Exception as e:
        return {
            "error": f"Error in standardization: {str(e)}"
        }
