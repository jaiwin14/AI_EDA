"""Data handling utilities for the EDA application."""
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime
from utils.serialization_fixed import to_json_serializable

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

        # Data preview with safe serialization
        preview_df = df.head()
        preview_data = []
        for _, row in preview_df.iterrows():
            row_dict = {}
            for col in preview_df.columns:
                val = row[col]
                if (pd.isna(val) if not hasattr(pd.isna(val), '__len__') else pd.isna(val).any()) or (isinstance(val, float) and np.isnan(val)):
                    row_dict[str(col)] = None
                else:
                    row_dict[str(col)] = to_json_serializable(val)
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
                    col_stats[stat] = to_json_serializable(val)
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
