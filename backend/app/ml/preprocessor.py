"""
Intelligent Preprocessing Module

This module handles automatic preprocessing of datasets using smart heuristics
and AI assistance when needed.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import json
import pandas as pd
import numpy as np
import logging
import warnings
from sklearn.impute import SimpleImputer, KNNImputer
from category_encoders import (
    CatBoostEncoder, WOEEncoder, GLMMEncoder, MEstimateEncoder,
    BinaryEncoder, CountEncoder, HashingEncoder, LeaveOneOutEncoder
)
from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler, 
    OneHotEncoder, 
    LabelEncoder,
    RobustScaler,
    PowerTransformer,
    QuantileTransformer,
    KBinsDiscretizer,
    PolynomialFeatures,
    FunctionTransformer,
    TargetEncoder,
    OrdinalEncoder
)
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    RFE,
    SelectFromModel
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders as ce
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import logging

from app.ml.gemini_client import GeminiClient  # Assuming you have this configured

logger = logging.getLogger(__name__)

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
                # Apply log(x+1) to non-negative numeric columns
                num_cols = X_transformed.select_dtypes(include=[np.number]).columns
                for col in num_cols:
                    if (X_transformed[col] >= 0).all() and X_transformed[col].nunique() > 1:
                        X_transformed[f'{col}_log'] = np.log1p(X_transformed[col])
                        
            elif op == 'square':
                # Square numeric features
                num_cols = X_transformed.select_dtypes(include=[np.number]).columns
                for col in num_cols:
                    X_transformed[f'{col}_squared'] = X_transformed[col] ** 2
                    
            elif op == 'interaction':
                # Create interaction terms for numeric features
                num_cols = X_transformed.select_dtypes(include=[np.number]).columns
                for i, col1 in enumerate(num_cols):
                    for col2 in num_cols[i+1:]:
                        X_transformed[f'{col1}_x_{col2}'] = X_transformed[col1] * X_transformed[col2]
                        
            elif op == 'date_features':
                # Extract date features from datetime columns
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


class IntelligentPreprocessor:
    """
    Advanced preprocessing pipeline with intelligent feature engineering,
    encoding, and transformation capabilities.
    """
    
    ENCODING_METHODS = [
        'onehot', 'label', 'ordinal', 'target', 'count', 'leave_one_out',
        'binary', 'hashing', 'woe', 'catboost', 'glmm', 'm_estimate'
    ]
    
    SCALING_METHODS = [
        'standard', 'minmax', 'robust', 'maxabs', 'power', 'quantile',
        'normalize', 'boxcox', 'yeo-johnson'
    ]
    
    FEATURE_ENGINEERING_OPS = [
        'log_transform', 'square', 'sqrt', 'interaction',
        'polynomial', 'date_features', 'bins'
    ]
    
    def __init__(self, 
                 use_gemini: bool = True,
                 max_categories: int = 20,
                 n_components: Optional[float] = 0.95,
                 feature_selection: bool = True):
        """Initialize the preprocessor.
        
        Args:
            use_gemini: Whether to use Gemini API for decision making
            max_categories: Maximum number of categories for one-hot encoding
            n_components: Percentage of variance to retain in PCA (None to disable)
            feature_selection: Whether to perform feature selection
        """
        self.use_gemini = use_gemini
        self.gemini = GeminiClient() if use_gemini else None
        self.max_categories = max_categories
        self.n_components = n_components
        self.feature_selection = feature_selection
        self.preprocessing_steps = {}
        self.feature_importances_ = None
        self.column_transformer = None
        self.feature_engineering_ops = []
        self.encoders = {}
        self.scalers = {}
        
    def _get_encoder(self, method: str, **kwargs):
        """Get the appropriate encoder based on the method."""
        if method == 'onehot':
            return OneHotEncoder(handle_unknown='ignore', sparse_output=False, **kwargs)
        elif method == 'label':
            return LabelEncoder()
        elif method == 'ordinal':
            return OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, **kwargs)
        elif method == 'target':
            return TargetEncoder(**kwargs)
        elif method == 'count':
            return ce.CountEncoder(**kwargs)
        elif method == 'leave_one_out':
            return ce.LeaveOneOutEncoder(**kwargs)
        elif method == 'binary':
            return ce.BinaryEncoder(**kwargs)
        elif method == 'hashing':
            return ce.HashingEncoder(**kwargs)
        elif method == 'woe':
            return ce.WOEEncoder(**kwargs)
        elif method == 'catboost':
            return ce.CatBoostEncoder(**kwargs)
        elif method == 'glmm':
            return ce.GLMMEncoder(**kwargs)
        elif method == 'm_estimate':
            return ce.MEstimateEncoder(**kwargs)
        else:
            return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
    def _get_scaler(self, method: str, **kwargs):
        """Get the appropriate scaler based on the method."""
        if method == 'standard':
            return StandardScaler(**kwargs)
        elif method == 'minmax':
            return MinMaxScaler(**kwargs)
        elif method == 'robust':
            return RobustScaler(**kwargs)
        elif method == 'maxabs':
            return MaxAbsScaler(**kwargs)
        elif method == 'power':
            return PowerTransformer(method='yeo-johnson', **kwargs)
        elif method == 'quantile':
            return QuantileTransformer(output_distribution='normal', **kwargs)
        elif method == 'boxcox':
            return PowerTransformer(method='box-cox', **kwargs)
        elif method == 'yeo-johnson':
            return PowerTransformer(method='yeo-johnson', **kwargs)
        else:
            return StandardScaler(**kwargs)
            
    def _get_feature_selector(self, method: str, n_features: int, task_type: str = 'classification'):
        """Get the appropriate feature selector based on the method."""
        if method == 'selectkbest':
            if task_type == 'regression':
                return SelectKBest(score_func=f_regression, k=n_features)
            return SelectKBest(score_func=f_classif, k=n_features)
            
        elif method == 'selectpercentile':
            if task_type == 'regression':
                return SelectPercentile(score_func=f_regression, percentile=n_features)
            return SelectPercentile(score_func=f_classif, percentile=n_features)
            
        elif method == 'mutual_info':
            if task_type == 'regression':
                return SelectKBest(score_func=mutual_info_regression, k=n_features)
            return SelectKBest(score_func=mutual_info_classif, k=n_features)
            
        elif method == 'rfe':
            if task_type == 'regression':
                estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            return RFE(estimator=estimator, n_features_to_select=n_features)
            
        elif method == 'selectfrommodel':
            if task_type == 'regression':
                estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            return SelectFromModel(estimator, max_features=n_features)
            
        else:
            return SelectKBest(score_func=f_classif, k=n_features)
    
    def _get_ai_suggestion(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI suggestion for preprocessing steps using Gemini API.
        
        Args:
            prompt: The specific preprocessing question/decision
            context: Dictionary containing data statistics and characteristics
            
        Returns:
            Dictionary with AI recommendations
        """
        if not self.use_gemini or not self.gemini:
            return {}
            
        try:
            # Prepare the context for the AI
            context_str = "\n".join([f"- {k}: {v}" for k, v in context.items()])
            
            system_prompt = """You are an expert data scientist specializing in data preprocessing. 
            Your task is to recommend the best preprocessing steps based on the data characteristics.
            Always respond with a valid JSON object containing your recommendations.
            """
            
            # Format the example JSON structure as a string to avoid f-string issues
            example_json = '''{
                "recommendation": "brief explanation of the recommendation",
                "action": "specific action to take (e.g., 'impute', 'encode', 'scale')",
                "method": "specific method to use (e.g., 'mean', 'onehot', 'standard')",
                "parameters": {
                    "parameter1": "value1",
                    "parameter2": "value2"
                }
            }'''
            
            user_prompt = f"""
            Given the following data characteristics:
            {context_str}
            
            {prompt}
            
            Please provide your recommendation as a JSON object with the following structure:
            {example_json}
            """
            
            # Get response from Gemini
            response = self.gemini.generate_text(
                prompt=user_prompt,
                system_instruction=system_prompt,
                response_format="json"
            )
            
            # Parse and validate the response
            if not response:
                logger.warning("Empty response from Gemini API")
                return {}
                
            try:
                # Parse the JSON response
                result = json.loads(response)
                
                # Validate the response structure
                if not isinstance(result, dict) or 'action' not in result:
                    logger.warning(f"Invalid response format from Gemini: {response}")
                    return {}
                    
                return result
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse Gemini response: {e}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting AI suggestion: {str(e)}")
            return {}
    
    def detect_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Automatically detect column types.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping column names to their types
        """
        column_types = {}
        
        for col in df.columns:
            # Skip columns with too many unique values (likely IDs)
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.9:  # More than 90% unique values
                column_types[col] = 'id'
                continue
                
            # Check data type
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() == 2:
                    column_types[col] = 'binary'
                else:
                    column_types[col] = 'numeric'
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                column_types[col] = 'datetime'
            else:
                # For categorical, check cardinality
                if df[col].nunique() < 20:  # Arbitrary threshold
                    column_types[col] = 'categorical_low_cardinality'
                else:
                    column_types[col] = 'categorical_high_cardinality'
                    
        return column_types
    
    def handle_missing_values(self, 
                           df: pd.DataFrame, 
                           column_types: Dict[str, str],
                           target_col: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Intelligently handle missing values with enhanced strategies.
        
        Args:
            df: Input DataFrame
            column_types: Dictionary of column types
            target_col: Name of the target column (if any)
            
        Returns:
            Tuple of (processed DataFrame, preprocessing steps)
        """
        steps = {}
        
        for col in df.columns:
            if df[col].isna().sum() == 0:
                continue
                
            col_type = column_types.get(col, 'unknown')
            missing_ratio = df[col].isna().mean()
            
            # If too many missing values, consider dropping
            if missing_ratio > 0.7:  # More than 70% missing
                steps[col] = {'action': 'drop', 'reason': 'high_missing_ratio'}
                df = df.drop(columns=[col])
                continue
                
            # Prepare context for AI suggestion
            context = {
                'column': col,
                'type': col_type,
                'missing_ratio': missing_ratio,
                'unique_values': df[col].nunique(),
                'is_target': col == target_col,
                'dtype': str(df[col].dtype),
                'skewness': df[col].skew() if pd.api.types.is_numeric_dtype(df[col]) else None,
                'mean': df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else None,
                'median': df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else None,
                'mode': df[col].mode().iloc[0] if not df[col].empty else None,
                'is_sequential': self._is_sequential(df[col]) if pd.api.types.is_datetime64_any_dtype(df[col]) else None
            }
            
            # Get AI suggestion if enabled
            if self.use_gemini:
                suggestion = self._get_ai_suggestion(
                    "What's the best way to handle missing values for this column? "
                    "Respond with a JSON object with 'action' and 'method' keys. "
                    "Possible actions: 'impute', 'drop', 'flag'. "
                    "For imputation, possible methods: 'mean', 'median', 'mode', 'knn', 'forward_fill', 'backward_fill', 'interpolate'.",
                    context
                )
                
                if suggestion and 'action' in suggestion:
                    action = suggestion['action']
                    method = suggestion.get('method')
                    params = suggestion.get('parameters', {})
                else:
                    action, method, params = self._get_default_missing_value_strategy(df[col], col_type, missing_ratio, target_col)
            else:
                action, method, params = self._get_default_missing_value_strategy(df[col], col_type, missing_ratio, target_col)
            
            # Apply the chosen method
            if action == 'impute':
                df, step = self._apply_imputation(df, col, method, params, col_type)
                steps[col] = step
                
            elif action == 'drop':
                df = df.drop(columns=[col])
                steps[col] = {'action': 'drop', 'reason': params.get('reason', 'user_preference')}
                
            elif action == 'flag':
                flag_col = f"{col}_missing"
                df[flag_col] = df[col].isna().astype(int)
                fill_value = params.get('fill_value', 0 if col_type == 'numeric' else 'missing')
                df[col] = df[col].fillna(fill_value)
                steps[col] = {
                    'action': 'flag', 
                    'flag_column': flag_col,
                    'fill_value': fill_value
                }
        
        return df, steps
        
    def _is_sequential(self, series: pd.Series) -> bool:
        """Check if a series has sequential values (e.g., time series data)."""
        if not pd.api.types.is_datetime64_any_dtype(series):
            return False
            
        # Check if the time differences are roughly equal
        diffs = series.dropna().sort_values().diff().dropna()
        if len(diffs) < 2:
            return False
            
        # If the standard deviation is small relative to the mean, it's likely sequential
        return (diffs.std() / diffs.mean().abs()) < 0.1
        
    def _get_default_missing_value_strategy(self, 
                                         series: pd.Series, 
                                         col_type: str, 
                                         missing_ratio: float,
                                         target_col: Optional[str] = None) -> tuple:
        """Get default strategy for handling missing values."""
        if col_type == 'numeric':
            if missing_ratio < 0.05:  # Small amount of missing data
                if abs(series.skew()) > 1:  # Highly skewed
                    return 'impute', 'median', {}
                else:
                    return 'impute', 'mean', {}
            else:  # More missing data, use more sophisticated methods
                if len(series) < 1000:  # Small dataset, use KNN
                    return 'impute', 'knn', {'n_neighbors': min(5, len(series) - 1)}
                else:  # Large dataset, use median for efficiency
                    return 'impute', 'median', {}
                    
        elif col_type in ['categorical_low_cardinality', 'binary']:
            if missing_ratio < 0.1:  # Small amount of missing data
                return 'impute', 'mode', {}
            else:  # More missing data, create a new category
                return 'impute', 'constant', {'fill_value': 'missing'}
                
        elif col_type == 'datetime':
            if self._is_sequential(series):
                return 'impute', 'interpolate', {'method': 'time'}
            else:
                return 'impute', 'mode', {}
                
        else:  # For high cardinality or unknown types
            if missing_ratio < 0.3:
                return 'flag', None, {'fill_value': 'missing'}
            else:
                return 'drop', None, {'reason': 'high_missing_ratio'}
    
    def _apply_imputation(self, 
                        df: pd.DataFrame, 
                        col: str, 
                        method: str, 
                        params: dict,
                        col_type: str) -> tuple:
        """Apply the specified imputation method to a column."""
        if method == 'mean':
            imputer = SimpleImputer(strategy='mean')
            df[col] = imputer.fit_transform(df[[col]]).ravel()
            return df, {'action': 'impute', 'method': 'mean'}
            
        elif method == 'median':
            imputer = SimpleImputer(strategy='median')
            df[col] = imputer.fit_transform(df[[col]]).ravel()
            return df, {'action': 'impute', 'method': 'median'}
            
        elif method == 'mode':
            imputer = SimpleImputer(strategy='most_frequent')
            df[col] = imputer.fit_transform(df[[col]]).ravel()
            return df, {'action': 'impute', 'method': 'mode'}
            
        elif method == 'knn':
            n_neighbors = params.get('n_neighbors', 5)
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df[col] = imputer.fit_transform(df[[col]]).ravel()
            return df, {'action': 'impute', 'method': 'knn', 'n_neighbors': n_neighbors}
            
        elif method == 'forward_fill':
            df[col] = df[col].fillna(method='ffill')
            return df, {'action': 'impute', 'method': 'forward_fill'}
            
        elif method == 'backward_fill':
            df[col] = df[col].fillna(method='bfill')
            return df, {'action': 'impute', 'method': 'backward_fill'}
            
        elif method == 'interpolate':
            method = params.get('method', 'linear')
            df[col] = df[col].interpolate(method=method)
            return df, {'action': 'impute', 'method': 'interpolate', 'interpolation': method}
            
        elif method == 'constant':
            fill_value = params.get('fill_value', 0 if col_type == 'numeric' else 'missing')
            df[col] = df[col].fillna(fill_value)
            return df, {'action': 'impute', 'method': 'constant', 'fill_value': fill_value}
            
        else:  # Default to mean for numeric, mode for others
            if col_type == 'numeric':
                imputer = SimpleImputer(strategy='mean')
                df[col] = imputer.fit_transform(df[[col]]).ravel()
                return df, {'action': 'impute', 'method': 'mean', 'note': 'default strategy'}
            else:
                imputer = SimpleImputer(strategy='most_frequent')
                df[col] = imputer.fit_transform(df[[col]]).ravel()
                return df, {'action': 'impute', 'method': 'mode', 'note': 'default strategy'}
    
    def _is_ordinal(self, series: pd.Series) -> bool:
        """Check if a categorical series has an inherent order."""
        try:
            # Check for common ordinal patterns
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
    
    def _get_default_encoding_strategy(self, 
                                     series: pd.Series, 
                                     col_type: str, 
                                     n_unique: int,
                                     has_target: bool,
                                     task_type: Optional[str] = None) -> tuple:
        """Determine the default encoding strategy for a categorical feature."""
        # For binary features
        if n_unique == 2:
            return 'binary', {}
            
        # For low cardinality
        if n_unique <= self.max_categories:
            # Check if it's ordinal
            if self._is_ordinal(series):
                return 'ordinal', {}
                
            # For classification with target, consider target encoding
            if has_target and task_type == 'classification' and n_unique > 5:
                return 'target', {}
                
            # Default to one-hot encoding for low cardinality
            return 'onehot', {}
            
        # For high cardinality
        else:
            # For classification with target, use target or catboost encoding
            if has_target:
                if task_type == 'classification':
                    return 'catboost', {}
                else:
                    return 'target', {}
            
            # For features with very high cardinality and no target
            if n_unique > 100:
                return 'hashing', {'n_components': min(10, n_unique // 20)}
            else:
                return 'count', {}
    
    def _apply_encoding(self, 
                       df: pd.DataFrame, 
                       col: str, 
                       method: str, 
                       params: dict,
                       target_col: Optional[str] = None) -> tuple:
        """Apply the specified encoding method to a column."""
        if method == 'onehot':
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, **params)
            encoded = encoder.fit_transform(df[[col]])
            
            # Create new column names
            categories = encoder.categories_[0]
            new_cols = [f"{col}_{str(cat).lower().replace(' ', '_')}" for cat in categories]
            
            # Add the one-hot encoded columns
            df[new_cols] = encoded
            df = df.drop(columns=[col])
            
            return df, {
                'method': 'onehot',
                'n_categories': len(categories),
                'new_columns': new_cols
            }
            
        elif method == 'ordinal':
            # For ordinal encoding, sort categories if they appear to be ordered
            categories = sorted(df[col].dropna().unique())
            
            # If the column appears to be ordinal numeric (e.g., '1', '2', '3')
            try:
                numeric_cats = [float(c) for c in categories if str(c).replace('.', '').isdigit()]
                if len(numeric_cats) == len(categories):
                    categories = sorted(numeric_cats)
            except:
                pass
                
            encoder = OrdinalEncoder(
                categories=[categories],
                handle_unknown='use_encoded_value',
                unknown_value=-1,
                **params
            )
            
            df[col] = encoder.fit_transform(df[[col]]).astype(int)
            
            return df, {
                'method': 'ordinal',
                'categories': categories,
                'n_categories': len(categories)
            }
            
        elif method == 'target':
            if target_col is None:
                raise ValueError("Target encoding requires a target column")
                
            # Use leave-one-out for training data to avoid target leakage
            encoder = ce.LeaveOneOutEncoder(**params)
            df[col] = encoder.fit_transform(df[col], df[target_col])
            
            return df, {
                'method': 'target',
                'n_categories': df[col].nunique(),
                'target': target_col
            }
            
        elif method == 'count':
            encoder = ce.CountEncoder(**params)
            df[col] = encoder.fit_transform(df[col])
            
            return df, {
                'method': 'count',
                'n_categories': df[col].nunique()
            }
            
        elif method == 'binary':
            encoder = ce.BinaryEncoder(**params)
            result = encoder.fit_transform(df[col])
            
            # Add the binary encoded columns
            for c in result.columns:
                df[f"{col}_{c}"] = result[c]
                
            df = df.drop(columns=[col])
            
            return df, {
                'method': 'binary',
                'n_components': len(result.columns),
                'n_categories': 2
            }
            
        elif method == 'hashing':
            n_components = params.get('n_components', 8)
            encoder = ce.HashingEncoder(n_components=n_components, **params)
            hashed = encoder.fit_transform(df[col])
            
            # Add the hashed features
            for i in range(n_components):
                df[f"{col}_hash_{i}"] = hashed.iloc[:, i]
                
            df = df.drop(columns=[col])
            
            return df, {
                'method': 'hashing',
                'n_components': n_components,
                'n_categories': df[col].nunique()
            }
            
        elif method in ['woe', 'catboost', 'glmm', 'm_estimate']:
            if target_col is None:
                raise ValueError(f"{method} encoding requires a target column")
                
            # Map method to the appropriate encoder class
            encoder_map = {
                'woe': ce.WOEEncoder,
                'catboost': ce.CatBoostEncoder,
                'glmm': ce.GLMMEncoder,
                'm_estimate': ce.MEstimateEncoder
            }
            
            encoder = encoder_map[method](**params)
            df[col] = encoder.fit_transform(df[col], df[target_col])
            
            return df, {
                'method': method,
                'n_categories': df[col].nunique(),
                'target': target_col
            }
            
        else:  # Default to label encoding
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            
            return df, {
                'method': 'label',
                'n_categories': df[col].nunique()
            }
    
    def encode_features(self, 
                       df: pd.DataFrame, 
                       column_types: Dict[str, str],
                       target_col: Optional[str] = None,
                       task_type: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Intelligently encode categorical features with advanced strategies.
        
        Args:
            df: Input DataFrame
            column_types: Dictionary of column types
            target_col: Name of the target column (if any)
            task_type: Type of ML task ('classification' or 'regression')
            
        Returns:
            Tuple of (encoded DataFrame, encoding steps)
        """
        steps = {}
        df_encoded = df.copy()
        
        # Determine task type if not provided
        if task_type is None and target_col is not None:
            if df[target_col].nunique() <= 10:  # Arbitrary threshold for classification
                task_type = 'classification'
            else:
                task_type = 'regression'
        
        for col in df.columns:
            col_type = column_types.get(col)
            
            # Skip non-categorical columns and target column
            if col_type not in ['categorical_low_cardinality', 'categorical_high_cardinality', 'binary'] or col == target_col:
                continue
                
            n_unique = df[col].nunique()
            
            # Skip ID columns
            if n_unique == len(df):
                steps[col] = {'action': 'drop', 'reason': 'high_cardinality_id'}
                df_encoded = df_encoded.drop(columns=[col])
                continue
            
            # Prepare context for AI suggestion if enabled
            if self.use_gemini:
                context = {
                    'column': col,
                    'type': col_type,
                    'n_unique': n_unique,
                    'max_categories': self.max_categories,
                    'has_target': target_col is not None,
                    'task_type': task_type,
                    'sample_values': df[col].value_counts().head(5).to_dict(),
                    'is_ordinal': self._is_ordinal(df[col])
                }
                
                suggestion = self._get_ai_suggestion(
                    "What's the best encoding method for this categorical feature? "
                    "Consider the cardinality, target variable, and task type. "
                    "Respond with a JSON object with 'method' and 'parameters' keys. "
                    "Possible methods: " + ", ".join(self.ENCODING_METHODS),
                    context
                )
                
                if suggestion and 'method' in suggestion and suggestion['method'] in self.ENCODING_METHODS:
                    method = suggestion['method']
                    params = suggestion.get('parameters', {})
                else:
                    method, params = self._get_default_encoding_strategy(
                        df[col], col_type, n_unique, target_col is not None, task_type
                    )
                    
                # Apply the selected encoding method
                try:
                    df_encoded, step = self._apply_encoding(
                        df_encoded, col, method, params, target_col
                    )
                    steps[col] = step
                except Exception as e:
                    logger.warning(f"Error applying {method} encoding to column {col}: {str(e)}. "
                                 "Falling back to label encoding.")
                    # Fall back to label encoding
                    encoder = LabelEncoder()
                    df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
                    steps[col] = {
                        'method': 'label',
                        'n_categories': df_encoded[col].nunique(),
                        'error': str(e)
                    }
        
        return df_encoded, steps
    
    def _should_scale_feature(self, series: pd.Series) -> bool:
        """Determine if a feature should be scaled based on its distribution."""
        if series.nunique() <= 2:  # Binary features
            return False
            
        stats = {
            'std': series.std(),
            'mean': abs(series.mean()),
            'min': series.min(),
            'max': series.max(),
            'iqr': series.quantile(0.75) - series.quantile(0.25)
        }
        
        # Check for large value ranges or high variance
        value_range = stats['max'] - stats['min']
        if value_range == 0:
            return False
            
        cv = stats['std'] / (stats['mean'] + 1e-10)  # Coefficient of variation
        
        # Scaling is needed if:
        # 1. High coefficient of variation (> 1.0)
        # 2. Large value range relative to mean (> 10x mean)
        # 3. Outliers present (using IQR method)
        return (cv > 1.0 or 
                value_range > 10 * abs(stats['mean']) or
                (stats['max'] > stats['mean'] + 3 * stats['std']) or
                (stats['min'] < stats['mean'] - 3 * stats['std']))

    def _select_best_scaler(self, series: pd.Series) -> str:
        """Select the most appropriate scaler based on data characteristics."""
        stats = {
            'skewness': series.skew(),
            'kurtosis': series.kurtosis(),
            'iqr': series.quantile(0.75) - series.quantile(0.25),
            'has_negatives': (series < 0).any(),
            'has_zeros': (series == 0).any(),
            'is_sparse': (series == 0).mean() > 0.5  # More than 50% zeros
        }
        
        # Check for normality
        from scipy import stats
        _, p_value = stats.normaltest(series.dropna())
        is_normal = p_value > 0.05
        
        # Decision tree for scaler selection
        if stats['is_sparse']:
            return 'maxabs'  # Preserves sparsity
        elif abs(stats['skewness']) > 1:  # Highly skewed
            if not stats['has_negatives'] and not stats['has_zeros']:
                return 'boxcox'  # Only for positive values
            else:
                return 'yeo-johnson'  # Handles all value ranges
        elif stats['iqr'] == 0:  # No variation
            return 'none'
        elif is_normal:
            return 'standard'  # StandardScaler for normal distributions
        else:
            return 'robust'  # RobustScaler for non-normal distributions with outliers

    def _apply_scaling(self, series: pd.Series, method: str) -> Tuple[pd.Series, dict]:
        """Apply the specified scaling method to a series."""
        if method == 'standard':
            scaler = StandardScaler()
            scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
            return pd.Series(scaled, index=series.index), {}
            
        elif method == 'minmax':
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
            return pd.Series(scaled, index=series.index), {}
            
        elif method == 'robust':
            scaler = RobustScaler()
            scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
            return pd.Series(scaled, index=series.index), {}
            
        elif method == 'maxabs':
            scaler = MaxAbsScaler()
            scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
            return pd.Series(scaled, index=series.index), {}
            
        elif method == 'quantile':
            scaler = QuantileTransformer(output_distribution='normal', n_quantiles=min(1000, len(series)))
            scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
            return pd.Series(scaled, index=series.index), {}
            
        elif method == 'power' and (series > 0).all():
            scaler = PowerTransformer(method='yeo-johnson')
            scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
            return pd.Series(scaled, index=series.index), {}
            
        elif method == 'boxcox' and (series > 0).all():
            from scipy import stats
            scaled, lmbda = stats.boxcox(series + 1e-6)  # Add small constant for zeros
            return pd.Series(scaled, index=series.index), {'lambda': lmbda}
            
        elif method == 'yeo-johnson':
            scaler = PowerTransformer(method='yeo-johnson')
            scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
            return pd.Series(scaled, index=series.index), {}
            
        return series, {'method': 'none', 'reason': 'No scaling applied'}

    def scale_features(self, 
                     df: pd.DataFrame, 
                     column_types: Dict[str, str],
                     target_col: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Scale numerical features using advanced techniques.
        
        Applies different scaling methods based on data distribution and characteristics.
        
        Args:
            df: Input DataFrame
            column_types: Dictionary of column types
            target_col: Name of the target column (if any)
            
        Returns:
            Tuple of (scaled DataFrame, scaling steps)
        """
        steps = {}
        numeric_cols = [col for col, t in column_types.items() 
                       if t in ['numeric', 'binary'] and col in df.columns]
        
        if not numeric_cols:
            return df, {}
        
        # Process each numeric column
        for col in numeric_cols:
            series = df[col]
            
            # Skip columns with no variation
            if series.nunique() <= 1:
                steps[col] = {'method': 'none', 'reason': 'No variation'}
                continue
                
            # Check if scaling is beneficial
            if not self._should_scale_feature(series):
                steps[col] = {'method': 'none', 'reason': 'No scaling needed'}
                continue
            
            # Get AI suggestion if enabled
            if self.use_gemini:
                context = {
                    'column': col,
                    'statistics': {
                        'mean': series.mean(),
                        'std': series.std(),
                        'min': series.min(),
                        'max': series.max(),
                        'skewness': series.skew(),
                        'kurtosis': series.kurtosis(),
                        'n_unique': series.nunique(),
                        'n_missing': series.isna().sum(),
                        'has_negatives': (series < 0).any(),
                        'has_zeros': (series == 0).any()
                    }
                }
                
                suggestion = self._get_ai_suggestion(
                    "What's the best scaling/normalization method for this numerical feature? "
                    "Consider the distribution, presence of outliers, and value ranges. "
                    "Respond with a JSON object with 'method' and 'reason' keys. "
                    "Possible methods: 'standard', 'minmax', 'robust', 'maxabs', 'quantile', 'power', 'boxcox', 'yeo-johnson', 'none'.",
                    context
                )
                
                if suggestion and 'method' in suggestion:
                    method = suggestion['method']
                    reason = suggestion.get('reason', 'AI suggestion')
                else:
                    method = self._select_best_scaler(series)
                    reason = 'Auto-selected based on data distribution'
            else:
                method = self._select_best_scaler(series)
                reason = 'Auto-selected based on data distribution'
            
            # Apply the selected scaling method
            try:
                scaled_series, method_info = self._apply_scaling(series, method)
                df[col] = scaled_series
                
                steps[col] = {
                    'method': method,
                    'reason': reason,
                    **method_info
                }
                
            except Exception as e:
                logger.warning(f"Error applying {method} scaling to column {col}: {str(e)}")
                steps[col] = {
                    'method': 'none',
                    'reason': f'Error: {str(e)}',
                    'error': str(e)
                }
        
        return df, steps
    
    def _get_optimal_components(self, X: pd.DataFrame, method: str = 'pca', 
                              max_components: int = 50, variance_threshold: float = 0.95) -> int:
        """Determine the optimal number of components for dimensionality reduction."""
        n_samples, n_features = X.shape
        max_components = min(n_features, max_components)
        
        if method == 'pca':
            # Use PCA to find components explaining 95% of variance
            pca = PCA(n_components=min(n_samples, n_features, max_components))
            pca.fit(X)
            
            # Find number of components that explain at least variance_threshold of variance
            explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(explained_variance_ratio >= variance_threshold) + 1
            return max(1, min(n_components, max_components))
            
        elif method == 'umap':
            # For UMAP, use a heuristic based on the data size
            return min(10, max(2, int(np.sqrt(n_features))))
            
        elif method == 'tsne':
            # t-SNE is typically used for 2D or 3D visualization
            return min(3, n_features)
            
        return min(10, n_features)

    def _apply_dimensionality_reduction(self, X: pd.DataFrame, method: str, 
                                      n_components: int, **kwargs) -> Tuple[np.ndarray, dict]:
        """Apply the specified dimensionality reduction method."""
        if method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components, random_state=42, **kwargs)
            reduced_data = reducer.fit_transform(X)
            return reduced_data, {
                'explained_variance_ratio': reducer.explained_variance_ratio_.tolist(),
                'singular_values': reducer.singular_values_.tolist()
            }
            
        elif method == 'ica':
            from sklearn.decomposition import FastICA
            reducer = FastICA(n_components=n_components, random_state=42, **kwargs)
            reduced_data = reducer.fit_transform(X)
            return reduced_data, {}
            
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            # t-SNE can be slow, so we might want to use PCA first for large datasets
            if X.shape[1] > 50:
                pca = PCA(n_components=min(50, X.shape[1]))
                X_transformed = pca.fit_transform(X)
            else:
                X_transformed = X
                
            tsne = TSNE(n_components=n_components, random_state=42, **kwargs)
            reduced_data = tsne.fit_transform(X_transformed)
            return reduced_data, {'kl_divergence': tsne.kl_divergence_}
            
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=n_components, random_state=42, **kwargs)
                reduced_data = reducer.fit_transform(X)
                return reduced_data, {}
            except ImportError:
                logger.warning("UMAP not installed. Using PCA instead.")
                return self._apply_dimensionality_reduction(X, 'pca', n_components)
                
        elif method == 'feature_selection':
            from sklearn.feature_selection import SelectKBest, f_classif, f_regression
            
            if 'target' in kwargs and kwargs['target'] is not None:
                y = kwargs['target']
                if len(np.unique(y)) > 10:  # Regression
                    selector = SelectKBest(f_regression, k=n_components)
                else:  # Classification
                    selector = SelectKBest(f_classif, k=n_components)
                reduced_data = selector.fit_transform(X, y)
                selected_features = X.columns[selector.get_support()].tolist()
                return reduced_data, {'selected_features': selected_features}
            return X.values, {}
            
        return X.values, {}

    def reduce_dimensionality(self, 
                            df: pd.DataFrame, 
                            column_types: Dict[str, str],
                            target_col: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply advanced dimensionality reduction techniques.
        
        Supports multiple methods including PCA, t-SNE, UMAP, and feature selection.
        Automatically selects the best method based on data characteristics.
        
        Args:
            df: Input DataFrame
            column_types: Dictionary of column types
            target_col: Name of the target column (if any)
            
        Returns:
            Tuple of (transformed DataFrame, reduction steps)
        """
        steps = {}
        
        # Only consider numeric features for reduction
        numeric_cols = [col for col, t in column_types.items() 
                       if t in ['numeric', 'binary'] and col in df.columns]
        
        if len(numeric_cols) < 3:  # Not enough features to reduce
            return df, {}
        
        # Separate features and target
        X = df[numeric_cols].copy()
        y = df[target_col] if target_col and target_col in df.columns else None
        
        # Get AI suggestion if enabled
        reduction_method = None
        if self.use_gemini:
            context = {
                'num_features': len(X.columns),
                'num_samples': len(X),
                'has_target': y is not None,
                'target_type': column_types.get(target_col, 'unknown') if target_col else 'none',
                'feature_correlation': X.corr().values.tolist()
            }
            
            suggestion = self._get_ai_suggestion(
                "What's the best dimensionality reduction approach for this dataset? "
                "Consider the number of features, samples, and whether there's a target variable. "
                "Respond with a JSON object with 'method' (pca, ica, tsne, umap, feature_selection, none), "
                "'n_components' (int), and 'reason' (str) keys.",
                context
            )
            
            if suggestion and 'method' in suggestion and suggestion['method'] != 'none':
                reduction_method = suggestion['method']
                n_components = suggestion.get('n_components')
        
        # If no AI suggestion, use auto-selection
        if not reduction_method:
            if len(X.columns) > 20:  # High-dimensional data
                reduction_method = 'pca'
                n_components = self._get_optimal_components(X, 'pca')
            elif len(X.columns) > 10:  # Medium-dimensional data
                reduction_method = 'feature_selection'
                n_components = min(10, len(X.columns) // 2)
            else:
                return df, {}  # No reduction needed
        
        # Apply the chosen reduction method
        try:
            # Scale data before reduction
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply dimensionality reduction
            reduced_data, method_info = self._apply_dimensionality_reduction(
                X_scaled, 
                method=reduction_method,
                n_components=n_components,
                target=y
            )
            
            # Create new column names
            if reduction_method == 'feature_selection':
                new_columns = method_info.get('selected_features', [])
                reduced_df = df[new_columns].copy()
            else:
                new_columns = [f'{reduction_method}_{i+1}' for i in range(reduced_data.shape[1])]
                reduced_df = pd.DataFrame(reduced_data, columns=new_columns, index=df.index)
            
            # Keep non-numeric and target columns
            non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
            if non_numeric_cols:
                reduced_df = pd.concat([reduced_df, df[non_numeric_cols]], axis=1)
            
            # Store reduction details
            steps['dimensionality_reduction'] = {
                'method': reduction_method,
                'n_components': n_components,
                'original_columns': X.columns.tolist(),
                'new_columns': new_columns,
                **method_info
            }
            
            return reduced_df, steps
            
        except Exception as e:
            logger.warning(f"Error applying {reduction_method} dimensionality reduction: {str(e)}")
            return df, {'error': str(e)}
    
    def _should_apply_preprocessing_step(self, df: pd.DataFrame, step: str, context: dict) -> bool:
        """Determine if a preprocessing step should be applied based on data characteristics."""
        if step == 'missing_values':
            # Check if there are any missing values
            return df.isnull().any().any()
            
        elif step == 'encoding':
            # Check if there are categorical columns to encode
            categorical_cols = [col for col, t in context.get('column_types', {}).items() 
                              if t in ['categorical', 'binary'] and col in df.columns]
            return len(categorical_cols) > 0
            
        elif step == 'scaling':
            # Check if scaling would be beneficial
            numeric_cols = [col for col, t in context.get('column_types', {}).items() 
                          if t in ['numeric', 'binary'] and col in df.columns]
            if not numeric_cols:
                return False
                
            # Check if any numeric column would benefit from scaling
            for col in numeric_cols:
                if self._should_scale_feature(df[col].dropna()):
                    return True
            return False
            
        elif step == 'dimensionality_reduction':
            # Only consider if we have enough features
            numeric_cols = [col for col, t in context.get('column_types', {}).items() 
                          if t in ['numeric', 'binary'] and col in df.columns]
            return len(numeric_cols) >= 5  # Only reduce if we have 5+ numeric features
            
        return False

    def _get_data_characteristics(self, df: pd.DataFrame, column_types: dict) -> dict:
        """Compute various data characteristics to guide preprocessing decisions."""
        stats = {
            'num_samples': len(df),
            'num_features': len(df.columns),
            'num_numeric': sum(1 for t in column_types.values() if t in ['numeric']),
            'num_categorical': sum(1 for t in column_types.values() if t in ['categorical', 'binary']),
            'has_missing': df.isnull().any().any(),
            'high_cardinality_features': [
                col for col, t in column_types.items() 
                if t in ['categorical'] and df[col].nunique() > 20
            ]
        }
        return stats

    def preprocess(self, 
                  df: pd.DataFrame, 
                  target_col: Optional[str] = None,
                  task_type: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply all preprocessing steps intelligently, only when beneficial.
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column (if any)
            task_type: Type of ML task ('classification', 'regression', None for auto-detect)
            
        Returns:
            Tuple of (preprocessed DataFrame, preprocessing steps)
        """
        if df.empty:
            return df, {}
        
        # Make a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Store all preprocessing steps
        preprocessing_steps = {}
        
        # 1. Detect column types
        column_types = self.detect_column_types(df_processed)
        preprocessing_steps['column_types'] = column_types
        
        # Get data characteristics to guide preprocessing decisions
        data_stats = self._get_data_characteristics(df_processed, column_types)
        
        # Create context for AI suggestions if enabled
        context = {
            'column_types': column_types,
            'data_stats': data_stats,
            'target_col': target_col,
            'task_type': task_type
        }
        
        # 2. Handle missing values (only if needed)
        if self._should_apply_preprocessing_step(df_processed, 'missing_values', context):
            df_processed, missing_steps = self.handle_missing_values(df_processed, column_types, target_col)
            if any(step.get('action') != 'none' for step in missing_steps.values()):
                preprocessing_steps['missing_values'] = missing_steps
        
        # 3. Encode categorical features (only if needed)
        if self._should_apply_preprocessing_step(df_processed, 'encoding', context):
            df_processed, encoding_steps = self.encode_features(df_processed, column_types, target_col, task_type)
            if encoding_steps:
                preprocessing_steps['encoding'] = encoding_steps
        
        # 4. Scale features (only if beneficial)
        if self._should_apply_preprocessing_step(df_processed, 'scaling', context):
            df_processed, scaling_steps = self.scale_features(df_processed, column_types, target_col)
            if scaling_steps:
                preprocessing_steps['scaling'] = scaling_steps
        
        # 5. Apply dimensionality reduction (only if beneficial)
        if self._should_apply_preprocessing_step(df_processed, 'dimensionality_reduction', context):
            df_processed, reduction_steps = self.reduce_dimensionality(df_processed, column_types, target_col)
            if reduction_steps and 'error' not in reduction_steps:
                preprocessing_steps['dimensionality_reduction'] = reduction_steps
        
        # Update the preprocessing steps
        self.preprocessing_steps = preprocessing_steps
        
        # Log the preprocessing steps taken
        if preprocessing_steps:
            logger.info(f"Applied preprocessing steps: {list(preprocessing_steps.keys())}")
        else:
            logger.info("No beneficial preprocessing steps were applied")
        
        return df_processed, preprocessing_steps
