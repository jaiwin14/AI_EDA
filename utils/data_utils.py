"""Data handling utilities for the EDA application."""
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime

def prepare_dataframe_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a comprehensive summary of a DataFrame with proper serialization handling.
    """
    try:
        # Basic information
        info = {
            "shape": list(df.shape),
            "columns": list(map(str, df.columns)),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
            "memory_usage": int(df.memory_usage(deep=True).sum())
        }

        # Missing values analysis
        missing_analysis = {
            "total_missing": int(df.isnull().sum().sum()),
            "missing_by_column": df.isnull().sum().to_dict(),
            "missing_percentage": df.isnull().sum().div(len(df)).mul(100).round(2).to_dict()
        }

        # Data preview with direct primitive type conversion
        preview_df = df.head()
        preview_data = []
        for _, row in preview_df.iterrows():
            row_dict = {}
            for col in preview_df.columns:
                val = row[col]
                if (pd.isna(val) if not hasattr(pd.isna(val), '__len__') else pd.isna(val).any()) or (isinstance(val, float) and np.isnan(val)):
                    row_dict[str(col)] = None
                else:
                    # Convert to primitive types directly
                    if isinstance(val, (int, float, bool)):
                        row_dict[str(col)] = val
                    elif isinstance(val, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                        row_dict[str(col)] = int(val)
                    elif isinstance(val, (np.floating, np.float64, np.float32, np.float16)):
                        row_dict[str(col)] = float(val)
                    elif isinstance(val, (np.bool_)):
                        row_dict[str(col)] = bool(val)
                    elif isinstance(val, (datetime, pd.Timestamp)):
                        row_dict[str(col)] = val.isoformat()
                    else:
                        row_dict[str(col)] = str(val)
            preview_data.append(row_dict)

        info["data_preview"] = preview_data

        # Statistical description
        desc_df = df.describe(include='all')
        description = {}
        for col in desc_df.columns:
            col_stats = {}
            for stat in desc_df.index:
                val = desc_df[col][stat]
                if (pd.isna(val) if not hasattr(pd.isna(val), '__len__') else pd.isna(val).any()) or (isinstance(val, float) and np.isnan(val)):
                    col_stats[stat] = None
                else:
                    # Convert to primitive types directly
                    if isinstance(val, (int, float, bool)):
                        col_stats[stat] = val
                    elif isinstance(val, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                        col_stats[stat] = int(val)
                    elif isinstance(val, (np.floating, np.float64, np.float32, np.float16)):
                        col_stats[stat] = float(val)
                    elif isinstance(val, (np.bool_)):
                        col_stats[stat] = bool(val)
                    elif isinstance(val, (datetime, pd.Timestamp)):
                        col_stats[stat] = val.isoformat()
                    else:
                        col_stats[stat] = str(val)
            description[str(col)] = col_stats

        # Column type categorization
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

        # Compile everything into a summary
        summary = {
            "info": info,
            "describe": description,
            "missing_analysis": missing_analysis,
            "column_types": {
                "numeric": numeric_cols,
                "categorical": categorical_cols,
                "datetime": datetime_cols
            },
            "generated_at": datetime.now().isoformat()
        }

        return summary

    except Exception as e:
        print(f"Error generating DataFrame summary: {str(e)}")
        return {
            "error": str(e),
            "message": "Failed to generate complete summary"
        }
