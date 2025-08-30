import pandas as pd
import numpy as np
import plotly.express as px
import base64
from io import BytesIO

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
    
    # Convert plot to base64 string
    buffer = BytesIO()
    fig.write_image(buffer, format="png")
    buffer.seek(0)
    
    return {
        "text": "Here's the correlation heatmap for numeric columns:",
        "plot": base64.b64encode(buffer.getvalue()).decode(),
        "data": corr.to_dict()
    }
