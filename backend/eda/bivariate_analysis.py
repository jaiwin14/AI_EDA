"""Bivariate Analysis Module with Interactive Visualizations"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Tuple
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.ai_utils import generate_insight
from utils.serialization_fixed import to_json_serializable

def run(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform bivariate analysis with appropriate visualizations for different data type combinations
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing bivariate analysis results and visualizations
    """
    try:
        results = {
            "text": f"Bivariate analysis completed for {len(df.columns)} columns",
            "pairwise_analyses": {},
            "summary": {},
            "ai_insights": ""
        }
        
        # Get column types
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Generate pairwise combinations
        all_columns = df.columns.tolist()
        pairwise_combinations = []
        
        for i in range(len(all_columns)):
            for j in range(i + 1, len(all_columns)):
                col1, col2 = all_columns[i], all_columns[j]
                pairwise_combinations.append((col1, col2))
        
        # Analyze each pair
        for col1, col2 in pairwise_combinations:
            pair_key = f"{col1}_vs_{col2}"
            analysis = analyze_pair(df[col1], df[col2], col1, col2)
            if analysis:
                results["pairwise_analyses"][pair_key] = analysis
        
        # Generate summary
        results["summary"] = generate_bivariate_summary(df, results["pairwise_analyses"])
        
        # Generate AI insights
        insight_prompt = f"""
        Analyze the bivariate analysis results for this dataset:
        - Total columns: {len(df.columns)}
        - Numeric columns: {len(numeric_columns)}
        - Categorical columns: {len(categorical_columns)}
        - Datetime columns: {len(datetime_columns)}
        - Total pairwise analyses: {len(results['pairwise_analyses'])}
        
        Provide insights on:
        1. Key relationships discovered between variables
        2. Patterns and trends in the data
        3. Potential correlations or associations
        4. Recommendations for further analysis
        5. Data quality observations from bivariate analysis
        """
        
        try:
            results["ai_insights"] = generate_insight(insight_prompt, context=results["pairwise_analyses"])
        except Exception as e:
            results["ai_insights"] = f"Unable to generate AI insights: {str(e)}"
        
        return results
        
    except Exception as e:
        return {
            "text": f"Error in bivariate analysis: {str(e)}",
            "error": str(e)
        }

def analyze_pair(series1: pd.Series, series2: pd.Series, col1: str, col2: str) -> Dict[str, Any]:
    """Analyze relationship between two columns"""
    
    # Remove rows where either column has missing values
    valid_mask = series1.notna() & series2.notna()
    if valid_mask.sum() < 10:  # Need at least 10 valid pairs
        return None
    
    clean_series1 = series1[valid_mask]
    clean_series2 = series2[valid_mask]
    
    # Determine analysis type based on data types
    type1 = get_column_type(series1)
    type2 = get_column_type(series2)
    
    analysis = {
        "column1": col1,
        "column2": col2,
        "type1": type1,
        "type2": type2,
        "analysis_type": f"{type1}_vs_{type2}",
        "valid_pairs": int(valid_mask.sum()),
        "missing_pairs": int((~valid_mask).sum())
    }
    
    # Perform appropriate analysis based on types
    if type1 == "numeric" and type2 == "numeric":
        analysis.update(analyze_numeric_vs_numeric(clean_series1, clean_series2))
    elif type1 == "categorical" and type2 == "categorical":
        analysis.update(analyze_categorical_vs_categorical(clean_series1, clean_series2))
    elif (type1 == "numeric" and type2 == "categorical") or (type1 == "categorical" and type2 == "numeric"):
        if type1 == "categorical":
            analysis.update(analyze_categorical_vs_numeric(clean_series1, clean_series2))
        else:
            analysis.update(analyze_categorical_vs_numeric(clean_series2, clean_series1))
    elif type1 == "datetime" or type2 == "datetime":
        analysis.update(analyze_datetime_relationship(clean_series1, clean_series2, type1, type2))
    
    return analysis

def get_column_type(series: pd.Series) -> str:
    """Determine the type of a column"""
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    elif pd.api.types.is_datetime64_dtype(series):
        return "datetime"
    else:
        return "categorical"

def analyze_numeric_vs_numeric(series1: pd.Series, series2: pd.Series) -> Dict[str, Any]:
    """Analyze relationship between two numeric columns"""
    
    # Correlation analysis
    correlation = float(series1.corr(series2))
    
    # Statistical measures
    stats = {
        "correlation": correlation,
        "correlation_interpretation": interpret_correlation(correlation),
        "covariance": float(series1.cov(series2)),
        "series1_mean": float(series1.mean()),
        "series1_std": float(series1.std()),
        "series2_mean": float(series2.mean()),
        "series2_std": float(series2.std())
    }
    
    # Create visualizations
    visualizations = {}
    
    # Scatter plot
    scatter_fig = go.Figure(data=go.Scatter(
        x=series1,
        y=series2,
        mode='markers',
        marker=dict(
            size=8,
            opacity=0.6,
            color=series1,  # Color by first variable
            colorscale='Viridis',
            showscale=True
        ),
        name="Scatter Plot"
    ))
    scatter_fig.update_layout(
        title=f"Scatter Plot: {series1.name} vs {series2.name}",
        xaxis_title=series1.name,
        yaxis_title=series2.name,
        height=400
    )
    visualizations["scatter"] = scatter_fig.to_dict()
    
    # Hexbin plot for large datasets
    if len(series1) > 1000:
        hexbin_fig = go.Figure(data=go.Histogram2d(
            x=series1,
            y=series2,
            nbinsx=50,
            nbinsy=50,
            colorscale='Viridis'
        ))
        hexbin_fig.update_layout(
            title=f"2D Histogram: {series1.name} vs {series2.name}",
            xaxis_title=series1.name,
            yaxis_title=series2.name,
            height=400
        )
        visualizations["hexbin"] = hexbin_fig.to_dict()
    
    # Joint distribution
    joint_fig = go.Figure()
    
    # Add marginal histograms
    joint_fig.add_trace(go.Histogram(
        x=series1,
        name=f"{series1.name} Distribution",
        yaxis="y2"
    ))
    joint_fig.add_trace(go.Histogram(
        y=series2,
        name=f"{series2.name} Distribution",
        xaxis="x2"
    ))
    
    joint_fig.update_layout(
        title=f"Joint Distribution: {series1.name} vs {series2.name}",
        xaxis2=dict(domain=[0.8, 1], anchor="y2"),
        yaxis2=dict(domain=[0.8, 1], anchor="x2"),
        height=500
    )
    visualizations["joint_distribution"] = joint_fig.to_dict()
    
    return {
        "statistics": stats,
        "visualizations": visualizations
    }

def analyze_categorical_vs_categorical(series1: pd.Series, series2: pd.Series) -> Dict[str, Any]:
    """Analyze relationship between two categorical columns"""
    
    # Contingency table
    contingency_table = pd.crosstab(series1, series2)
    
    # Chi-square test
    try:
        from scipy.stats import chi2_contingency
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        chi2_result = {
            "chi2_statistic": float(chi2),
            "p_value": float(p_value),
            "degrees_of_freedom": int(dof),
            "significant": p_value < 0.05
        }
    except ImportError:
        chi2_result = {"error": "scipy not available for chi-square test"}
    
    # Cramer's V (measure of association)
    try:
        n = len(series1)
        min_dim = min(contingency_table.shape) - 1
        cramer_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        chi2_result["cramer_v"] = float(cramer_v)
    except:
        chi2_result["cramer_v"] = None
    
    # Create visualizations
    visualizations = {}
    
    # Heatmap of contingency table
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=contingency_table.values,
        x=contingency_table.columns,
        y=contingency_table.index,
        colorscale='Viridis',
        text=contingency_table.values,
        texttemplate="%{text}",
        textfont={"size": 10},
        showscale=True
    ))
    heatmap_fig.update_layout(
        title=f"Contingency Table: {series1.name} vs {series2.name}",
        xaxis_title=series2.name,
        yaxis_title=series1.name,
        height=400
    )
    visualizations["heatmap"] = heatmap_fig.to_dict()
    
    # Stacked bar chart
    stacked_fig = go.Figure()
    
    for category in series1.unique():
        subset = series2[series1 == category]
        value_counts = subset.value_counts()
        stacked_fig.add_trace(go.Bar(
            name=str(category),
            x=value_counts.index,
            y=value_counts.values
        ))
    
    stacked_fig.update_layout(
        title=f"Stacked Bar Chart: {series1.name} vs {series2.name}",
        xaxis_title=series2.name,
        yaxis_title="Count",
        barmode='stack',
        height=400
    )
    visualizations["stacked_bar"] = stacked_fig.to_dict()
    
    return {
        "contingency_table": contingency_table.to_dict(),
        "chi2_test": chi2_result,
        "visualizations": visualizations
    }

def analyze_categorical_vs_numeric(cat_series: pd.Series, num_series: pd.Series) -> Dict[str, Any]:
    """Analyze relationship between categorical and numeric columns"""
    
    # Group statistics
    group_stats = num_series.groupby(cat_series).agg(['count', 'mean', 'std', 'min', 'max']).round(2)
    
    # ANOVA test
    try:
        from scipy.stats import f_oneway
        groups = [group for name, group in num_series.groupby(cat_series)]
        f_stat, p_value = f_oneway(*groups)
        anova_result = {
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05
        }
    except ImportError:
        anova_result = {"error": "scipy not available for ANOVA test"}
    
    # Create visualizations
    visualizations = {}
    
    # Box plot
    box_fig = go.Figure()
    
    for category in cat_series.unique():
        subset = num_series[cat_series == category]
        box_fig.add_trace(go.Box(
            y=subset,
            name=str(category),
            boxpoints='outliers'
        ))
    
    box_fig.update_layout(
        title=f"Box Plot: {num_series.name} by {cat_series.name}",
        yaxis_title=num_series.name,
        xaxis_title=cat_series.name,
        height=400
    )
    visualizations["boxplot"] = box_fig.to_dict()
    
    # Violin plot
    violin_fig = go.Figure()
    
    for category in cat_series.unique():
        subset = num_series[cat_series == category]
        violin_fig.add_trace(go.Violin(
            y=subset,
            name=str(category),
            box_visible=True,
            meanline_visible=True
        ))
    
    violin_fig.update_layout(
        title=f"Violin Plot: {num_series.name} by {cat_series.name}",
        yaxis_title=num_series.name,
        xaxis_title=cat_series.name,
        height=400
    )
    visualizations["violin"] = violin_fig.to_dict()
    
    # Bar chart of means
    means = num_series.groupby(cat_series).mean()
    bar_fig = go.Figure(data=go.Bar(
        x=means.index,
        y=means.values,
        name="Mean Values"
    ))
    bar_fig.update_layout(
        title=f"Mean {num_series.name} by {cat_series.name}",
        xaxis_title=cat_series.name,
        yaxis_title=f"Mean {num_series.name}",
        height=400
    )
    visualizations["mean_bar"] = bar_fig.to_dict()
    
    return {
        "group_statistics": group_stats.to_dict(),
        "anova_test": anova_result,
        "visualizations": visualizations
    }

def analyze_datetime_relationship(series1: pd.Series, series2: pd.Series, type1: str, type2: str) -> Dict[str, Any]:
    """Analyze relationship involving datetime columns"""
    
    # For now, focus on datetime vs numeric or categorical
    if type1 == "datetime":
        dt_series, other_series = series1, series2
        other_type = type2
    else:
        dt_series, other_series = series2, series1
        other_type = type1
    
    # Extract time components
    time_components = {
        "year": dt_series.dt.year,
        "month": dt_series.dt.month,
        "day": dt_series.dt.day,
        "dayofweek": dt_series.dt.dayofweek,
        "hour": dt_series.dt.hour if hasattr(dt_series.dt, 'hour') else None
    }
    
    # Create visualizations
    visualizations = {}
    
    if other_type == "numeric":
        # Time series plot
        ts_fig = go.Figure(data=go.Scatter(
            x=dt_series,
            y=other_series,
            mode='lines+markers',
            name="Time Series"
        ))
        ts_fig.update_layout(
            title=f"Time Series: {other_series.name} over time",
            xaxis_title="Time",
            yaxis_title=other_series.name,
            height=400
        )
        visualizations["timeseries"] = ts_fig.to_dict()
        
        # Monthly/Yearly aggregation
        monthly_avg = other_series.groupby(dt_series.dt.to_period('M')).mean()
        monthly_fig = go.Figure(data=go.Bar(
            x=monthly_avg.index.astype(str),
            y=monthly_avg.values,
            name="Monthly Average"
        ))
        monthly_fig.update_layout(
            title=f"Monthly Average {other_series.name}",
            xaxis_title="Month",
            yaxis_title=f"Average {other_series.name}",
            height=400
        )
        visualizations["monthly_avg"] = monthly_fig.to_dict()
    
    return {
        "time_components": {k: v.to_dict() if v is not None else None for k, v in time_components.items()},
        "visualizations": visualizations
    }

def interpret_correlation(correlation: float) -> str:
    """Interpret correlation coefficient"""
    abs_corr = abs(correlation)
    if abs_corr < 0.1:
        return "Negligible"
    elif abs_corr < 0.3:
        return "Weak"
    elif abs_corr < 0.5:
        return "Moderate"
    elif abs_corr < 0.7:
        return "Strong"
    else:
        return "Very Strong"

def generate_bivariate_summary(df: pd.DataFrame, pairwise_analyses: Dict) -> Dict[str, Any]:
    """Generate summary of bivariate analyses"""
    
    summary = {
        "total_pairs": len(pairwise_analyses),
        "analysis_types": {},
        "strong_correlations": [],
        "significant_relationships": []
    }
    
    # Count analysis types
    for analysis in pairwise_analyses.values():
        analysis_type = analysis.get("analysis_type", "unknown")
        summary["analysis_types"][analysis_type] = summary["analysis_types"].get(analysis_type, 0) + 1
    
    # Find strong correlations
    for pair_key, analysis in pairwise_analyses.items():
        if "statistics" in analysis and "correlation" in analysis["statistics"]:
            corr = analysis["statistics"]["correlation"]
            if abs(corr) > 0.7:
                summary["strong_correlations"].append({
                    "pair": pair_key,
                    "correlation": corr,
                    "interpretation": analysis["statistics"]["correlation_interpretation"]
                })
    
    # Find significant relationships (chi-square, ANOVA)
    for pair_key, analysis in pairwise_analyses.items():
        if "chi2_test" in analysis and analysis["chi2_test"].get("significant", False):
            summary["significant_relationships"].append({
                "pair": pair_key,
                "test": "chi-square",
                "p_value": analysis["chi2_test"]["p_value"]
            })
        elif "anova_test" in analysis and analysis["anova_test"].get("significant", False):
            summary["significant_relationships"].append({
                "pair": pair_key,
                "test": "ANOVA",
                "p_value": analysis["anova_test"]["p_value"]
            })
    
    return summary
