import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.metrics import mutual_info_score
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

def calculate_mutual_information(X: pd.DataFrame, y: pd.Series, 
                               task_type: Optional[str] = None) -> pd.Series:
    """Calculate mutual information between features and target"""
    try:
        if task_type is None:
            if y.dtype == 'object' or y.nunique() <= min(20, len(y) * 0.05):
                task_type = 'classification'
            else:
                task_type = 'regression'
        
        X_processed = _prepare_features_for_mi(X)
        
        if task_type == 'classification':
            if y.dtype == 'object':
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
            else:
                y_encoded = y
            mi_scores = mutual_info_classif(X_processed, y_encoded, random_state=42)
        else:
            mi_scores = mutual_info_regression(X_processed, y, random_state=42)
        
        return pd.Series(mi_scores, index=X.columns)
        
    except Exception as e:
        logger.warning(f"MI calculation failed: {str(e)}")
        return pd.Series(0.0, index=X.columns)

def _prepare_features_for_mi(X: pd.DataFrame) -> np.ndarray:
    """Prepare features for mutual information calculation"""
    X_processed = X.copy()
    for col in X_processed.columns:
        if X_processed[col].dtype == 'object':
            le = LabelEncoder()
            mask = X_processed[col].notna()
            if mask.sum() > 0:
                X_processed.loc[mask, col] = le.fit_transform(X_processed.loc[mask, col])
                X_processed.loc[~mask, col] = -1
            else:
                X_processed[col] = -1
        elif X_processed[col].dtype in ['int64', 'float64']:
            X_processed[col] = X_processed[col].fillna(X_processed[col].median())
        X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce').fillna(0)
    return X_processed.values

def calculate_correlation_matrix(df: pd.DataFrame, method: str = 'pearson') -> Dict[str, Any]:
    """Calculate correlation matrix for numeric columns"""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {'matrix': pd.DataFrame(), 'strong_pairs': [], 'method': method}
        
        corr_matrix = df[numeric_cols].corr(method=method)
        strong_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= 0.7:
                    strong_pairs.append({
                        'feature_1': corr_matrix.columns[i],
                        'feature_2': corr_matrix.columns[j],
                        'correlation': corr_val,
                        'abs_correlation': abs(corr_val)
                    })
        strong_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
        return {'matrix': corr_matrix, 'strong_pairs': strong_pairs, 'method': method}
    except Exception as e:
        logger.warning(f"Correlation calculation failed: {str(e)}")
        return {'matrix': pd.DataFrame(), 'strong_pairs': [], 'method': method}

def calculate_cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Calculate Cramér's V for categorical variables"""
    try:
        contingency_table = pd.crosstab(x, y)
        chi2, _, _, _ = chi2_contingency(contingency_table)
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        if min_dim == 0:
            return 0.0
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        return cramers_v
    except Exception as e:
        logger.warning(f"Cramér's V calculation failed: {str(e)}")
        return 0.0

def calculate_categorical_associations(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate associations between categorical variables"""
    try:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(cat_cols) < 2:
            return {'cramers_v_matrix': pd.DataFrame(), 'strong_associations': []}
        
        cramers_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols)
        for i, col1 in enumerate(cat_cols):
            for j, col2 in enumerate(cat_cols):
                if i == j:
                    cramers_matrix.loc[col1, col2] = 1.0
                elif i < j:
                    cramers_v = calculate_cramers_v(df[col1], df[col2])
                    cramers_matrix.loc[col1, col2] = cramers_v
                    cramers_matrix.loc[col2, col1] = cramers_v
        cramers_matrix = cramers_matrix.astype(float)
        
        strong_associations = []
        for i in range(len(cat_cols)):
            for j in range(i+1, len(cat_cols)):
                cramers_v = cramers_matrix.iloc[i, j]
                if cramers_v >= 0.5:
                    strong_associations.append({
                        'variable_1': cat_cols[i],
                        'variable_2': cat_cols[j],
                        'cramers_v': cramers_v
                    })
        strong_associations.sort(key=lambda x: x['cramers_v'], reverse=True)
        return {'cramers_v_matrix': cramers_matrix, 'strong_associations': strong_associations}
    except Exception as e:
        logger.warning(f"Categorical associations calculation failed: {str(e)}")
        return {'cramers_v_matrix': pd.DataFrame(), 'strong_associations': []}

def calculate_numeric_categorical_relationships(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate relationships between numeric and categorical variables using ANOVA"""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not numeric_cols or not cat_cols:
            return {'anova_results': [], 'significant_relationships': []}
        
        anova_results = []
        significant_relationships = []
        for num_col in numeric_cols:
            for cat_col in cat_cols:
                try:
                    data = df[[num_col, cat_col]].dropna()
                    if len(data) < 10:
                        continue
                    groups = [group[num_col].values for name, group in data.groupby(cat_col)]
                    groups = [group for group in groups if len(group) > 0]
                    if len(groups) < 2:
                        continue
                    f_stat, p_value = f_oneway(*groups)
                    result = {
                        'numeric_variable': num_col,
                        'categorical_variable': cat_col,
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                    anova_results.append(result)
                    if p_value < 0.05:
                        significant_relationships.append(result)
                except Exception as e:
                    logger.warning(f"ANOVA failed for {num_col} vs {cat_col}: {str(e)}")
                    continue
        significant_relationships.sort(key=lambda x: x['f_statistic'], reverse=True)
        return {'anova_results': anova_results, 'significant_relationships': significant_relationships}
    except Exception as e:
        logger.warning(f"Numeric-categorical relationships calculation failed: {str(e)}")
        return {'anova_results': [], 'significant_relationships': []}

def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Variance Inflation Factor for multicollinearity detection"""
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return pd.DataFrame()
        numeric_df = numeric_df.loc[:, numeric_df.var() > 0]
        if numeric_df.shape[1] < 2:
            return pd.DataFrame()
        numeric_df = numeric_df.fillna(numeric_df.median())
        vif_data = pd.DataFrame()
        vif_data["Feature"] = numeric_df.columns
        vif_data["VIF"] = [variance_inflation_factor(numeric_df.values, i) 
                          for i in range(len(numeric_df.columns))]
        vif_data = vif_data.sort_values('VIF', ascending=False)
        return vif_data
    except ImportError:
        logger.warning("statsmodels not available for VIF calculation")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"VIF calculation failed: {str(e)}")
        return pd.DataFrame()

def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> Dict[str, Any]:
    """Detect outliers using IQR method"""
    try:
        if not pd.api.types.is_numeric_dtype(series):
            return {'outlier_indices': [], 'outlier_count': 0, 'outlier_percentage': 0.0, 'bounds': (None, None)}
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_indices = series[outlier_mask].index.tolist()
        outlier_count = len(outlier_indices)
        outlier_percentage = (outlier_count / len(series)) * 100 if len(series) > 0 else 0
        return {
            'outlier_indices': outlier_indices,
            'outlier_count': outlier_count,
            'outlier_percentage': outlier_percentage,
            'bounds': (lower_bound, upper_bound),
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR
        }
    except Exception as e:
        logger.warning(f"Outlier detection failed: {str(e)}")
        return {'outlier_indices': [], 'outlier_count': 0, 'outlier_percentage': 0.0, 'bounds': (None, None)}

def calculate_feature_stability(df: pd.DataFrame, n_samples: int = 1000, 
                               n_iterations: int = 10) -> Dict[str, float]:
    """Calculate feature stability by measuring variance in statistics across subsamples"""
    try:
        stability_scores = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                means = []
                for _ in range(n_iterations):
                    if len(df) >= n_samples:
                        sample = df[col].sample(n=n_samples, random_state=np.random.randint(0, 10000))
                    else:
                        sample = df[col]
                    means.append(sample.mean())
                if len(means) > 1 and np.mean(means) != 0:
                    cv = np.std(means) / abs(np.mean(means))
                    stability_scores[col] = max(0, 1 - cv)
                else:
                    stability_scores[col] = 1.0
            else:
                mode_frequencies = []
                for _ in range(n_iterations):
                    if len(df) >= n_samples:
                        sample = df[col].sample(n=n_samples, random_state=np.random.randint(0, 10000))
                    else:
                        sample = df[col]
                    mode_freq = sample.value_counts().iloc[0] / len(sample) if len(sample) > 0 else 0
                    mode_frequencies.append(mode_freq)
                if len(mode_frequencies) > 0:
                    stability_scores[col] = 1 - np.std(mode_frequencies)
                else:
                    stability_scores[col] = 0.0
        return stability_scores
    except Exception as e:
        logger.warning(f"Feature stability calculation failed: {str(e)}")
        return {col: 0.5 for col in df.columns}

# ✅ New function added
def calculate_anova_f_score(df: pd.DataFrame, numeric_col: str, categorical_col: str) -> float:
    """
    Calculate ANOVA F-score between a numeric and a categorical variable.
    
    Args:
        df (pd.DataFrame): Input dataframe
        numeric_col (str): Name of the numeric column
        categorical_col (str): Name of the categorical column
    
    Returns:
        float: ANOVA F-statistic. Returns 0.0 if calculation fails.
    """
    try:
        data = df[[numeric_col, categorical_col]].dropna()
        groups = [group[numeric_col].values for _, group in data.groupby(categorical_col)]
        if len(groups) < 2:
            return 0.0
        f_stat, _ = f_oneway(*groups)
        return f_stat
    except Exception:
        return 0.0
