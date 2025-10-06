from typing import Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer


def is_sequential(series: pd.Series) -> bool:
    if not pd.api.types.is_datetime64_any_dtype(series):
        return False
    diffs = series.dropna().sort_values().diff().dropna()
    if len(diffs) < 2:
        return False
    return (diffs.std() / diffs.mean().abs()) < 0.1


def default_missing_strategy(series: pd.Series, col_type: str, missing_ratio: float,
                             target_col: Optional[str] = None) -> tuple:
    if col_type == 'numeric':
        if missing_ratio < 0.05:
            return ('impute', 'median', {}) if abs(series.skew()) > 1 else ('impute', 'mean', {})
        if len(series) < 1000:
            return 'impute', 'knn', {'n_neighbors': min(5, max(2, len(series) - 1))}
        return 'impute', 'median', {}
    if col_type in ['categorical_low_cardinality', 'binary']:
        return ('impute', 'mode', {}) if missing_ratio < 0.1 else ('impute', 'constant', {'fill_value': 'missing'})
    if col_type == 'datetime':
        return ('impute', 'interpolate', {'method': 'time'}) if is_sequential(series) else ('impute', 'mode', {})
    return ('flag', None, {'fill_value': 'missing'}) if missing_ratio < 0.3 else ('drop', None, {'reason': 'high_missing_ratio'})


def apply_imputation(df: pd.DataFrame, col: str, method: str, params: Dict[str, Any], col_type: str) -> tuple:
    if method == 'mean':
        imputer = SimpleImputer(strategy='mean')
        df[col] = imputer.fit_transform(df[[col]]).ravel()
        return df, {'action': 'impute', 'method': 'mean'}
    if method == 'median':
        imputer = SimpleImputer(strategy='median')
        df[col] = imputer.fit_transform(df[[col]]).ravel()
        return df, {'action': 'impute', 'method': 'median'}
    if method == 'mode':
        imputer = SimpleImputer(strategy='most_frequent')
        df[col] = imputer.fit_transform(df[[col]]).ravel()
        return df, {'action': 'impute', 'method': 'mode'}
    if method == 'knn':
        n_neighbors = params.get('n_neighbors', 5)
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df[col] = imputer.fit_transform(df[[col]]).ravel()
        return df, {'action': 'impute', 'method': 'knn', 'n_neighbors': n_neighbors}
    if method == 'forward_fill':
        df[col] = df[col].fillna(method='ffill')
        return df, {'action': 'impute', 'method': 'forward_fill'}
    if method == 'backward_fill':
        df[col] = df[col].fillna(method='bfill')
        return df, {'action': 'impute', 'method': 'backward_fill'}
    if method == 'interpolate':
        m = params.get('method', 'linear')
        df[col] = df[col].interpolate(method=m)
        return df, {'action': 'impute', 'method': 'interpolate', 'interpolation': m}
    if method == 'constant':
        fill_value = params.get('fill_value', 0 if col_type == 'numeric' else 'missing')
        df[col] = df[col].fillna(fill_value)
        return df, {'action': 'impute', 'method': 'constant', 'fill_value': fill_value}
    if col_type == 'numeric':
        imputer = SimpleImputer(strategy='mean')
        df[col] = imputer.fit_transform(df[[col]]).ravel()
        return df, {'action': 'impute', 'method': 'mean', 'note': 'default strategy'}
    imputer = SimpleImputer(strategy='most_frequent')
    df[col] = imputer.fit_transform(df[[col]]).ravel()
    return df, {'action': 'impute', 'method': 'mode', 'note': 'default strategy'}


def handle_missing_values(df: pd.DataFrame, column_types: Dict[str, str],
                          ai_suggester=None, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    steps: Dict[str, Any] = {}
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        col_type = column_types.get(col, 'unknown')
        missing_ratio = df[col].isna().mean()
        if missing_ratio > 0.7:
            steps[col] = {'action': 'drop', 'reason': 'high_missing_ratio'}
            df = df.drop(columns=[col])
            continue
        context = {
            'column': col,
            'type': col_type,
            'missing_ratio': missing_ratio,
            'unique_values': df[col].nunique(),
            'is_target': col == target_col,
            'dtype': str(df[col].dtype)
        }
        action = method = None
        params: Dict[str, Any] = {}
        if ai_suggester is not None:
            suggestion = ai_suggester.get_ai_suggestion(
                "What's the best way to handle missing values for this column?",
                context
            )
            if suggestion and 'action' in suggestion:
                action = suggestion['action']
                method = suggestion.get('method')
                params = suggestion.get('parameters', {})
        if action is None:
            action, method, params = default_missing_strategy(df[col], col_type, missing_ratio, target_col)
        if action == 'impute':
            df, step = apply_imputation(df, col, method, params, col_type)
            steps[col] = step
        elif action == 'drop':
            df = df.drop(columns=[col])
            steps[col] = {'action': 'drop', 'reason': params.get('reason', 'user_preference')}
        elif action == 'flag':
            flag_col = f"{col}_missing"
            df[flag_col] = df[col].isna().astype(int)
            fill_value = params.get('fill_value', 0 if col_type == 'numeric' else 'missing')
            df[col] = df[col].fillna(fill_value)
            steps[col] = {'action': 'flag', 'flag_column': flag_col, 'fill_value': fill_value}
    return df, steps


