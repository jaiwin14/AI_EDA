"""Dataset Information and Describe Analysis Module"""
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
    Generate comprehensive dataset information and describe statistics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing dataset information and describe statistics
    """
    try:
        # Basic dataset information
        dataset_info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.apply(str).to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "total_cells": df.shape[0] * df.shape[1],
            "missing_cells": df.isnull().sum().sum(),
            "missing_percentage": round((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2)
        }
        
        # Detailed describe statistics
        describe_stats = df.describe(include='all')
        describe_dict = to_json_serializable(describe_stats)
        
        # Column type breakdown
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        column_types = {
            "numeric": numeric_columns,
            "categorical": categorical_columns,
            "datetime": datetime_columns,
            "total_numeric": len(numeric_columns),
            "total_categorical": len(categorical_columns),
            "total_datetime": len(datetime_columns)
        }
        
        # Missing values by column
        missing_by_column = df.isnull().sum().to_dict()
        missing_percentage_by_column = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        
        # Unique values by column
        unique_values = {}
        for col in df.columns:
            unique_count = df[col].nunique()
            unique_values[col] = {
                "count": int(unique_count),
                "percentage": round((unique_count / len(df)) * 100, 2)
            }
        
        # Generate AI insights
        insight_prompt = f"""
        Analyze this dataset information:
        - Dataset shape: {df.shape}
        - Column types: {column_types}
        - Missing values: {dataset_info['missing_percentage']}% of total cells
        - Memory usage: {dataset_info['memory_usage_mb']} MB
        
        Provide insights on:
        1. Data quality assessment
        2. Potential data type conversions needed
        3. Missing value patterns
        4. Recommendations for preprocessing
        5. Overall dataset characteristics
        """
        
        try:
            ai_insights = generate_insight(insight_prompt, context={
                "dataset_info": dataset_info,
                "column_types": column_types,
                "missing_by_column": missing_by_column
            })
        except Exception as e:
            ai_insights = f"Unable to generate AI insights: {str(e)}"
        
        # Create visualizations
        visualizations = create_dataset_visualizations(df, column_types, missing_by_column)
        
        return {
            "text": f"Dataset Information Analysis completed for {df.shape[0]} rows and {df.shape[1]} columns",
            "dataset_info": dataset_info,
            "describe_stats": describe_dict,
            "column_types": column_types,
            "missing_by_column": missing_by_column,
            "missing_percentage_by_column": missing_percentage_by_column,
            "unique_values": unique_values,
            "ai_insights": ai_insights,
            "visualizations": visualizations
        }
        
    except Exception as e:
        return {
            "text": f"Error in dataset information analysis: {str(e)}",
            "error": str(e)
        }

def create_dataset_visualizations(df: pd.DataFrame, column_types: Dict, missing_by_column: Dict) -> Dict[str, Any]:
    """Create visualizations for dataset overview"""
    visualizations = {}
    
    try:
        # Missing values heatmap
        missing_data = df.isnull()
        missing_fig = go.Figure(data=go.Heatmap(
            z=missing_data.values,
            x=df.columns,
            y=list(range(len(df))),
            colorscale='Reds',
            showscale=True
        ))
        missing_fig.update_layout(
            title="Missing Values Heatmap",
            xaxis_title="Columns",
            yaxis_title="Rows",
            height=400
        )
        visualizations["missing_heatmap"] = missing_fig.to_dict()
        
        # Column types distribution
        type_counts = {
            "Numeric": len(column_types["numeric"]),
            "Categorical": len(column_types["categorical"]),
            "Datetime": len(column_types["datetime"])
        }
        
        type_fig = go.Figure(data=go.Pie(
            labels=list(type_counts.keys()),
            values=list(type_counts.values()),
            hole=0.3
        ))
        type_fig.update_layout(
            title="Column Types Distribution",
            height=400
        )
        visualizations["column_types"] = type_fig.to_dict()
        
        # Missing values by column bar chart
        missing_fig_bar = go.Figure(data=go.Bar(
            x=list(missing_by_column.keys()),
            y=list(missing_by_column.values()),
            marker_color='red'
        ))
        missing_fig_bar.update_layout(
            title="Missing Values by Column",
            xaxis_title="Columns",
            yaxis_title="Missing Count",
            height=400
        )
        visualizations["missing_bar"] = missing_fig_bar.to_dict()
        
    except Exception as e:
        visualizations["error"] = f"Error creating visualizations: {str(e)}"
    
    return visualizations
