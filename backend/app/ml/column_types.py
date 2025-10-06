from typing import Dict
import pandas as pd


def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    column_types: Dict[str, str] = {}
    for col in df.columns:
        unique_ratio = df[col].nunique() / len(df) if len(df) else 0
        if unique_ratio > 0.9:
            column_types[col] = 'id'
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            column_types[col] = 'binary' if df[col].nunique() == 2 else 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types[col] = 'datetime'
        else:
            column_types[col] = 'categorical_low_cardinality' if df[col].nunique() < 20 else 'categorical_high_cardinality'
    return column_types


