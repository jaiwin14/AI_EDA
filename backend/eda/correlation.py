import pandas as pd
import numpy as np
import plotly.express as px
from .utils import export_plot_as_image

def run(df: pd.DataFrame) -> dict:
    """Generate correlation heatmap for numeric columns"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return {
            "text": "No numeric columns found for correlation analysis",
            "plot": None
        }
    
    corr = numeric_df.corr()
    
    # Create heatmap using plotly
    fig = px.imshow(
        corr,
        labels=dict(color="Correlation"),
        title="Correlation Heatmap"
    )
    
    # Convert plot to base64 string using utility function
    return {
        "text": "Here's the correlation heatmap for numeric columns:",
        "plot": export_plot_as_image(fig, width=800, height=600),
        "data": corr.to_dict()
    }
