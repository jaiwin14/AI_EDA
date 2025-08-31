import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .utils import export_plot_as_image

def run(df: pd.DataFrame) -> dict:
    """Generate distribution plots for numeric columns"""
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return {
            "text": "No numeric columns found for distribution analysis",
            "plot": None
        }
    
    # Limit to first 6 columns for better visualization
    cols_to_plot = numeric_df.columns[:min(6, len(numeric_df.columns))]
    
    # Create subplots
    fig = make_subplots(rows=len(cols_to_plot), cols=1, 
                       subplot_titles=[f'Distribution of {col}' for col in cols_to_plot],
                       vertical_spacing=0.05)
    
    # Add histograms for each column
    for i, col in enumerate(cols_to_plot):
        fig.add_trace(
            go.Histogram(x=numeric_df[col], name=col, nbinsx=30),
            row=i+1, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=300 * len(cols_to_plot),
        width=800,
        title_text="Data Distribution Analysis",
        showlegend=False
    )
    
    # Convert plot to base64 string using utility function
    plot_height = 300 * len(cols_to_plot)
    
    # Calculate basic statistics for each column
    stats = {}
    for col in cols_to_plot:
        stats[col] = {
            "mean": float(numeric_df[col].mean()),
            "median": float(numeric_df[col].median()),
            "std": float(numeric_df[col].std()),
            "min": float(numeric_df[col].min()),
            "max": float(numeric_df[col].max()),
            "skewness": float(numeric_df[col].skew())
        }
    
    return {
        "text": "Here's the distribution analysis for numeric columns:",
        "plot": export_plot_as_image(fig, width=800, height=plot_height),
        "stats": stats
    }