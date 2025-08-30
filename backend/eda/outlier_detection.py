import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO

def run(df: pd.DataFrame) -> dict:
    """Detect and visualize outliers in the dataset"""
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return {
            "text": "No numeric columns found for outlier detection",
            "plot": None
        }
    
    # Limit to first 6 columns for better visualization
    cols_to_plot = numeric_df.columns[:min(6, len(numeric_df.columns))]
    
    # Create subplots - one boxplot for each column
    fig = make_subplots(rows=len(cols_to_plot), cols=1, 
                       subplot_titles=[f'Boxplot of {col}' for col in cols_to_plot],
                       vertical_spacing=0.05)
    
    # Add boxplots for each column
    outlier_stats = {}
    for i, col in enumerate(cols_to_plot):
        fig.add_trace(
            go.Box(y=numeric_df[col], name=col, boxmean=True),
            row=i+1, col=1
        )
        
        # Calculate outlier statistics using IQR method
        q1 = numeric_df[col].quantile(0.25)
        q3 = numeric_df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = numeric_df[(numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)][col]
        
        outlier_stats[col] = {
            "count": len(outliers),
            "percentage": (len(outliers) / len(numeric_df)) * 100,
            "min_value": float(outliers.min()) if not outliers.empty else None,
            "max_value": float(outliers.max()) if not outliers.empty else None,
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound)
        }
    
    # Update layout
    fig.update_layout(
        height=300 * len(cols_to_plot),
        width=800,
        title_text="Outlier Detection Analysis",
        showlegend=False
    )
    
    # Convert plot to base64 string
    buffer = BytesIO()
    fig.write_image(buffer, format="png")
    buffer.seek(0)
    
    # Create a summary of outliers
    total_outliers = sum(stats["count"] for stats in outlier_stats.values())
    
    return {
        "text": f"Outlier analysis complete. Found {total_outliers} potential outliers across {len(cols_to_plot)} numeric columns.",
        "plot": base64.b64encode(buffer.getvalue()).decode(),
        "stats": outlier_stats
    }