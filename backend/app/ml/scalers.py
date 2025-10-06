from typing import Dict, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer, Normalizer


def should_scale(series: pd.Series) -> bool:
    if series.nunique() <= 2:
        return False
    stats = {
        'std': series.std(),
        'mean': abs(series.mean()),
        'min': series.min(),
        'max': series.max()
    }
    value_range = stats['max'] - stats['min']
    if value_range == 0:
        return False
    cv = stats['std'] / (stats['mean'] + 1e-10)
    return (cv > 1.0 or value_range > 10 * abs(stats['mean']) or
            (stats['max'] > stats['mean'] + 3 * stats['std']) or
            (stats['min'] < stats['mean'] - 3 * stats['std']))


def select_best_scaler(series: pd.Series) -> str:
    s = {
        'skewness': series.skew(),
        'iqr': series.quantile(0.75) - series.quantile(0.25),
        'has_negatives': (series < 0).any(),
        'has_zeros': (series == 0).any(),
        'is_sparse': (series == 0).mean() > 0.5
    }
    try:
        from scipy import stats as spstats
        _, p_value = spstats.normaltest(series.dropna())
        is_normal = p_value > 0.05
    except Exception:
        is_normal = False
    if s['is_sparse']:
        return 'maxabs'
    if abs(s['skewness']) > 1:
        return 'boxcox' if (not s['has_negatives'] and not s['has_zeros']) else 'yeo-johnson'
    if s['iqr'] == 0:
        return 'none'
    return 'standard' if is_normal else 'robust'


def apply_scaling(series: pd.Series, method: str) -> Tuple[pd.Series, Dict]:
    if method == 'standard':
        scaler = StandardScaler()
        scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
        return pd.Series(scaled, index=series.index), {}
    if method == 'minmax':
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
        return pd.Series(scaled, index=series.index), {}
    if method == 'robust':
        scaler = RobustScaler()
        scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
        return pd.Series(scaled, index=series.index), {}
    if method == 'maxabs':
        scaler = MaxAbsScaler()
        scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
        return pd.Series(scaled, index=series.index), {}
    if method == 'quantile':
        scaler = QuantileTransformer(output_distribution='normal', n_quantiles=min(1000, len(series)))
        scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
        return pd.Series(scaled, index=series.index), {}
    if method == 'boxcox' and (series > 0).all():
        from scipy import stats as spstats
        scaled, lmbda = spstats.boxcox(series + 1e-6)
        return pd.Series(scaled, index=series.index), {'lambda': lmbda}
    if method == 'yeo-johnson':
        scaler = PowerTransformer(method='yeo-johnson')
        scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
        return pd.Series(scaled, index=series.index), {}
    if method == 'normalize':
        scaler = Normalizer()
        scaled = scaler.fit_transform(series.values.reshape(1, -1)).flatten()
        return pd.Series(scaled, index=series.index), {}
    return series, {'method': 'none', 'reason': 'No scaling applied'}


