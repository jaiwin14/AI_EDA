import pandas as pd
import plotly.express as px
import base64
from io import BytesIO

def run(df: pd.DataFrame) -> dict:
    """Analyze missing values in the dataset"""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Missing %': missing_pct.values
    })
    
    # Only keep columns with missing values
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    
    if missing_df.empty:
        return {
            "text": "No missing values found in the dataset!",
            "plot": None
        }
    
    # Create missing values plot
    fig = px.bar(
        missing_df,
        x='Column',
        y='Missing %',
        title='Missing Values Analysis'
    )
    
    # Convert plot to base64 string
    buffer = BytesIO()
    fig.write_image(buffer, format="png")
    buffer.seek(0)
    
    return {
        "text": "Here's the missing values analysis:",
        "plot": base64.b64encode(buffer.getvalue()).decode(),
        "data": missing_df.to_dict()
    }
