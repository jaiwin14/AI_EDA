from typing import Any, Dict, Optional, Tuple
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
import category_encoders as ce


def is_ordinal(series: pd.Series) -> bool:
    try:
        common_ordinals = [
            ['low', 'medium', 'high'],
            ['small', 'medium', 'large'],
            ['bad', 'average', 'good', 'excellent'],
            ['never', 'rarely', 'sometimes', 'often', 'always']
        ]
        unique_vals = set(str(v).lower() for v in series.unique() if pd.notna(v))
        for ordinal in common_ordinals:
            if all(val in unique_vals for val in ordinal):
                return True
        return False
    except Exception:
        return False


def default_encoding_strategy(series: pd.Series, col_type: str, n_unique: int,
                              has_target: bool, task_type: Optional[str], max_categories: int) -> tuple:
    if n_unique == 2:
        return 'binary', {}
    if n_unique <= max_categories:
        if is_ordinal(series):
            return 'ordinal', {}
        if has_target and task_type == 'classification' and n_unique > 5:
            return 'target', {}
        return 'onehot', {}
    if has_target:
        return ('catboost', {}) if task_type == 'classification' else ('target', {})
    if n_unique > 100:
        return 'hashing', {'n_components': min(10, n_unique // 20)}
    return 'count', {}


def apply_encoding(df: pd.DataFrame, col: str, method: str, params: Dict[str, Any],
                   target_col: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    n_unique_before = df[col].nunique()
    if method == 'onehot':
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, **params)
        encoded = encoder.fit_transform(df[[col]])
        categories = encoder.categories_[0]
        new_cols = [f"{col}_{str(cat).lower().replace(' ', '_')}" for cat in categories]
        df[new_cols] = encoded
        df = df.drop(columns=[col])
        return df, {'method': 'onehot', 'n_categories': len(categories), 'new_columns': new_cols}
    if method == 'ordinal':
        categories = sorted(df[col].dropna().unique())
        try:
            numeric_cats = [float(c) for c in categories if str(c).replace('.', '').isdigit()]
            if len(numeric_cats) == len(categories):
                categories = sorted(numeric_cats)
        except Exception:
            pass
        encoder = OrdinalEncoder(categories=[categories], handle_unknown='use_encoded_value', unknown_value=-1, **params)
        df[col] = encoder.fit_transform(df[[col]]).astype(int)
        return df, {'method': 'ordinal', 'categories': categories, 'n_categories': len(categories)}
    if method == 'target':
        if target_col is None:
            raise ValueError("Target encoding requires a target column")
        encoder = ce.LeaveOneOutEncoder(**params)
        df[col] = encoder.fit_transform(df[col], df[target_col])
        return df, {'method': 'target', 'n_categories': df[col].nunique(), 'target': target_col}
    if method == 'count':
        encoder = ce.CountEncoder(**params)
        df[col] = encoder.fit_transform(df[col])
        return df, {'method': 'count', 'n_categories': df[col].nunique()}
    if method == 'binary':
        encoder = ce.BinaryEncoder(**params)
        result = encoder.fit_transform(df[col])
        for c in result.columns:
            df[f"{col}_{c}"] = result[c]
        df = df.drop(columns=[col])
        return df, {'method': 'binary', 'n_components': len(result.columns), 'n_categories': 2}
    if method == 'hashing':
        n_components = params.get('n_components', 8)
        encoder = ce.HashingEncoder(n_components=n_components, **params)
        hashed = encoder.fit_transform(df[col])
        for i in range(n_components):
            df[f"{col}_hash_{i}"] = hashed.iloc[:, i]
        df = df.drop(columns=[col])
        return df, {'method': 'hashing', 'n_components': n_components, 'n_categories': n_unique_before}
    if method in ['woe', 'catboost', 'glmm', 'm_estimate']:
        if target_col is None:
            raise ValueError(f"{method} encoding requires a target column")
        encoder_map = {
            'woe': ce.WOEEncoder,
            'catboost': ce.CatBoostEncoder,
            'glmm': ce.GLMMEncoder,
            'm_estimate': ce.MEstimateEncoder
        }
        encoder = encoder_map[method](**params)
        df[col] = encoder.fit_transform(df[col], df[target_col])
        return df, {'method': method, 'n_categories': df[col].nunique(), 'target': target_col}
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col].astype(str))
    return df, {'method': 'label', 'n_categories': df[col].nunique()}


