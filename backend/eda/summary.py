import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
from io import BytesIO

def run(df: pd.DataFrame) -> dict:
    """Generate summary statistics for the dataframe"""
    summary = {
        "text": "Here are the summary statistics for your dataset:",
        "stats": {
            "shape": df.shape,
            "numeric_stats": df.describe().to_dict(),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
    }
    
    # Generate a summary table plot
    fig = go.Figure(data=[go.Table(
        header=dict(values=["Metric", "Value"]),
        cells=dict(values=[
            ["Rows", "Columns", "Memory Usage (MB)"],
            [df.shape[0], df.shape[1], f"{summary['stats']['memory_usage']:.2f}"]
        ])
    )])
    
    # Convert plot to base64 string
    buffer = BytesIO()
    fig.write_image(buffer, format="png")
    buffer.seek(0)
    summary["plot"] = base64.b64encode(buffer.getvalue()).decode()
    
    return summary
