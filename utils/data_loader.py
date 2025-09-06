"""Utility functions for loading and preparing data."""
import pandas as pd

def prepare_dataframe(df):
    """
    Prepare a dataframe for analysis by handling basic cleaning and type conversion.
    """
    # Convert column names to strings
    df.columns = df.columns.astype(str)
    
    # Handle any initial type conversions
    for col in df.columns:
        # Convert object types that might be numeric
        if df[col].dtype == 'object':
            try:
                numeric_series = pd.to_numeric(df[col])
                if not numeric_series.isna().all():  # If any conversion succeeded
                    df[col] = numeric_series
            except (ValueError, TypeError):
                # Keep original values if conversion fails
                continue
    
    return df

def create_dataframe_summary(df):
    """
    Create a summary of the dataframe including basic statistics and info.
    """
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.apply(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist()
    }
    return summary
