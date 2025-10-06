import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for feature engineering operations."""
    
    def __init__(self, operations: List[str] = None):
        self.operations = operations or []
        self.feature_names = []
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X_transformed = X.copy()
        
        for op in self.operations:
            if op == 'log_transform':
                num_cols = X_transformed.select_dtypes(include=[np.number]).columns
                for col in num_cols:
                    if (X_transformed[col] >= 0).all() and X_transformed[col].nunique() > 1:
                        X_transformed[f'{col}_log'] = np.log1p(X_transformed[col])
            elif op == 'square':
                num_cols = X_transformed.select_dtypes(include=[np.number]).columns
                for col in num_cols:
                    X_transformed[f'{col}_squared'] = X_transformed[col] ** 2
            elif op == 'interaction':
                num_cols = X_transformed.select_dtypes(include=[np.number]).columns
                for i, col1 in enumerate(num_cols):
                    for col2 in num_cols[i+1:]:
                        X_transformed[f'{col1}_x_{col2}'] = X_transformed[col1] * X_transformed[col2]
            elif op == 'date_features':
                date_cols = X_transformed.select_dtypes(include=['datetime64']).columns
                for col in date_cols:
                    X_transformed[f'{col}_year'] = X_transformed[col].dt.year
                    X_transformed[f'{col}_month'] = X_transformed[col].dt.month
                    X_transformed[f'{col}_day'] = X_transformed[col].dt.day
                    X_transformed[f'{col}_dayofweek'] = X_transformed[col].dt.dayofweek
                    X_transformed[f'{col}_is_weekend'] = X_transformed[col].dt.dayofweek.isin([5, 6]).astype(int)
        
        self.feature_names = X_transformed.columns.tolist()
        return X_transformed
    
    def get_feature_names(self):
        return self.feature_names


