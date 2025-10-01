"""
ML Model Training Pipeline
Automated model training with hyperparameter optimization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import uuid

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from app.core.config import settings

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Automated ML model training with preprocessing and evaluation"""
    
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.task_type = None
        self.target_column = None
        self.feature_columns = []
        
    async def _get_supported_tasks(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, bool]:
        """
        Determine which task types are supported by the dataset
        
        Args:
            df: Input DataFrame
            target_column: Optional target column name
            
        Returns:
            Dictionary mapping task types to boolean indicating if they're supported
        """
        tasks = {
            'classification': False,
            'regression': False,
            'clustering': True,  # Clustering is always possible (unsupervised)
            'time_series': False
        }
        
        # Check for time series (datetime index or columns)
        datetime_cols = [
            col for col in df.columns 
            if pd.api.types.is_datetime64_any_dtype(df[col]) 
            or str(df[col].dtype).startswith('datetime')
        ]
        tasks['time_series'] = len(datetime_cols) > 0 or isinstance(df.index, pd.DatetimeIndex)
        
        # If no target column, only clustering is possible
        if target_column is None or target_column not in df.columns:
            return tasks
            
        target_series = df[target_column]
        
        # Check for classification
        if target_series.dtype in ['object', 'category']:
            tasks['classification'] = True
        else:
            # Check if numeric target could be classification
            unique_ratio = target_series.nunique() / len(target_series)
            if unique_ratio < 0.05 or target_series.nunique() <= 20:
                tasks['classification'] = True
            
            # Check for regression
            if len(target_series) > 10 and target_series.dtype in ['int64', 'float64']:
                tasks['regression'] = True
                
        return tasks
        
    async def auto_train_models(self, df: pd.DataFrame, target_column: str = None, 
                              task_type: str = 'auto') -> Dict[str, Any]:
        """
        Automatically train models based on the data and task type.
        If task_type is 'auto', will test all applicable pipelines.
        
        Args:
            df: Input DataFrame
            target_column: Optional target column name
            task_type: Type of task ('classification', 'regression', 'clustering', 
                     'time_series', 'auto', or list of specific tasks to try)
            
        Returns:
            Dictionary containing training results for all applicable pipelines
        """
        try:
            # Store original data
            self.df = df.copy()
            
            # Get supported tasks for this dataset
            supported_tasks = await self._get_supported_tasks(df, target_column)
            
            # Handle different task_type inputs
            if task_type == 'auto':
                tasks_to_run = [t for t, supported in supported_tasks.items() if supported]
                if not tasks_to_run:
                    return {"status": "failed", "error": "No supported task types found for this dataset"}
            elif isinstance(task_type, str):
                if task_type not in supported_tasks or not supported_tasks[task_type]:
                    return {"status": "failed", "error": f"Task type '{task_type}' is not supported by this dataset"}
                tasks_to_run = [task_type]
            elif isinstance(task_type, list):
                tasks_to_run = [t for t in task_type if t in supported_tasks and supported_tasks[t]]
                if not tasks_to_run:
                    return {"status": "failed", "error": "None of the specified task types are supported by this dataset"}
            else:
                return {"status": "failed", "error": f"Invalid task_type: {task_type}"}
            
            logger.info(f"Running tasks: {', '.join(tasks_to_run)}")
            
            all_results = {}
            
            for current_task in tasks_to_run:
                self.task_type = current_task
                logger.info(f"\n=== Starting {current_task} pipeline ===")
                
                try:
                    # Prepare data for this task type
                    if target_column is not None and target_column in df.columns:
                        X = df.drop(columns=[target_column])
                        y = df[target_column]
                        
                        # Handle different split strategies
                        if current_task == 'time_series':
                            # Time series split
                            train_size = int(len(X) * 0.8)
                            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
                            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
                        else:
                            # Standard train/test split
                            if current_task == 'classification':
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.2, random_state=settings.RANDOM_STATE,
                                    stratify=y if len(np.unique(y)) > 1 and len(y) > 10 else None
                                )
                            else:  # regression or clustering
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.2, random_state=settings.RANDOM_STATE
                                )
                    else:
                        # For clustering, we don't have a target
                        X_train, X_test = train_test_split(
                            df, test_size=0.2, random_state=settings.RANDOM_STATE
                        )
                        y_train, y_test = None, None
                    
                    # Get models for this task
                    models = await self._get_models_for_task(current_task, X_train, y_train)
                    
                    # Train models
                    task_results = {}
                    for model_name, model_config in models.items():
                        try:
                            logger.info(f"Training {current_task} model: {model_name}...")
                            result = await self._train_single_model(
                                model_name, model_config, X_train, X_test, y_train, y_test
                            )
                            task_results[model_name] = result
                        except Exception as e:
                            logger.error(f"Error training {model_name}: {str(e)}")
                            task_results[model_name] = {
                                "status": "failed",
                                "error": str(e)
                            }
                    
                    # Store results for this task
                    all_results[current_task] = {
                        "status": "completed",
                        "models": task_results,
                        "task_type": current_task,
                        "target_column": target_column,
                        "feature_columns": X_train.columns.tolist() if hasattr(X_train, 'columns') else [],
                        "dataset_info": {
                            "n_samples": len(X_train) + (len(X_test) if X_test is not None else 0),
                            "n_features": X_train.shape[1] if hasattr(X_train, 'shape') else 0,
                            "n_classes": len(np.unique(y_train)) if y_train is not None and hasattr(y_train, '__iter__') else None,
                            "class_distribution": dict(pd.Series(y_train).value_counts().items()) if y_train is not None and hasattr(y_train, '__iter__') else None
                        },
                        "training_params": {
                            "test_size": 0.2,
                            "random_state": settings.RANDOM_STATE
                        },
                        "best_model": None,
                        "best_metric": None,
                        "metric_name": None,
                        "training_time_seconds": None,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    # Find best model for this task
                    valid_results = [r for r in task_results.values() if r.get('status') == 'completed']
                    if valid_results:
                        primary_metric = next(iter(valid_results[0]['metrics'].get('test', {}).keys()), None)
                        if primary_metric:
                            # Determine if higher or lower is better for this metric
                            higher_is_better = primary_metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'r2', 'silhouette']
                            
                            best_model = max(
                                valid_results,
                                key=lambda x: x['metrics']['test'].get(primary_metric, float('-inf') if higher_is_better else float('inf'))
                            )
                            
                            all_results[current_task].update({
                                "best_model": best_model['model_name'],
                                "best_model_id": best_model.get('model_id'),
                                "best_metric": best_model['metrics']['test'].get(primary_metric),
                                "metric_name": primary_metric
                            })
                    
                    logger.info(f"Completed {current_task} pipeline")
                    
                except Exception as e:
                    logger.error(f"Error in {current_task} pipeline: {str(e)}", exc_info=True)
                    all_results[current_task] = {
                        "status": "failed",
                        "error": str(e)
                    }
            
            # If we only ran one task, return its results directly
            if len(all_results) == 1:
                return {
                    "status": "completed",
                    "task_type": tasks_to_run[0],
                    **all_results[tasks_to_run[0]]
                }
            
            # Otherwise, return all results with a summary
            return {
                "status": "completed",
                "tasks_run": tasks_to_run,
                "results": all_results,
                "supported_tasks": supported_tasks
            }
            
        except Exception as e:
            logger.error(f"Error in auto_train_models: {str(e)}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _auto_detect_target(self, df: pd.DataFrame) -> Optional[str]:
        """Auto-detect target column using heuristics"""
        
        candidates = []
        
        for col in df.columns:
            score = 0
            
            # Check column name for target keywords
            target_keywords = ['target', 'label', 'class', 'outcome', 'result', 'prediction', 
                              'price', 'salary', 'income', 'survived', 'diagnosis', 'species']
            
            col_lower = col.lower()
            for keyword in target_keywords:
                if keyword in col_lower:
                    score += 30
                    break
            
            # Check data characteristics
            unique_ratio = df[col].nunique() / len(df)
            
            # Categorical with low cardinality (good for classification)
            if df[col].dtype in ['object', 'category'] and unique_ratio < 0.1:
                score += 20
            
            # Numeric with high cardinality (good for regression)
            elif pd.api.types.is_numeric_dtype(df[col]) and unique_ratio > 0.8:
                score += 15
            
            # Position bias - targets often at the end
            if df.columns.get_loc(col) >= len(df.columns) - 3:
                score += 10
            
            candidates.append((col, score))
        
        # Return column with highest score if above threshold
        candidates.sort(key=lambda x: x[1], reverse=True)
        if candidates[0][1] > 20:
            return candidates[0][0]
        
        return None
    
    async def _detect_task_type(self, df: pd.DataFrame, target_column: str = None) -> str:
        """
        Detect the type of machine learning task
        
        Args:
            df: Input dataframe
            target_column: Optional target column name (None for unsupervised)
            
        Returns:
            Task type: 'classification', 'regression', 'clustering', or 'time_series'
        """
        # If no target column, it's an unsupervised task
        if target_column is None or target_column not in df.columns:
            # Check if time series (has datetime index or column)
            datetime_cols = [col for col in df.columns if 
                           pd.api.types.is_datetime64_any_dtype(df[col]) or 
                           df[col].dtype in ['datetime64[ns]', 'datetime64[ms]']]
            
            if datetime_cols or isinstance(df.index, pd.DatetimeIndex):
                return 'time_series'
            return 'clustering'
            
        target_series = df[target_column]
        
        # Check data type for supervised tasks
        if target_series.dtype in ['object', 'category']:
            return 'classification'
        
        # Check unique values ratio for numeric columns
        unique_ratio = target_series.nunique() / len(target_series)
        
        if unique_ratio < 0.05 or target_series.nunique() <= 20:
            return 'classification'
        else:
            # Check if time series (temporal patterns in target)
            if len(target_series) > 10:
                from statsmodels.tsa.stattools import adfuller
                try:
                    # Test for stationarity
                    result = adfuller(target_series.dropna())
                    if result[1] > 0.05:  # Non-stationary time series
                        return 'time_series'
                except:
                    pass
            return 'regression'
    
    async def _prepare_features_target(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variables"""
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Handle target encoding for classification
        if self.task_type == 'classification':
            if y.dtype in ['object', 'category']:
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y), index=y.index)
                # Store label encoder for later use
                self.label_encoder = le
        
        return X, y
    
    async def _create_preprocessor(self, X_train: pd.DataFrame) -> ColumnTransformer:
        """Create preprocessing pipeline"""
        
        # Identify column types
        numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Numeric preprocessing
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical preprocessing
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return preprocessor
    
    async def _get_models_for_task(self, task_type: str, X: pd.DataFrame = None, y: pd.Series = None) -> Dict[str, Dict[str, Any]]:
        """
        Get appropriate models for the task type with intelligent selection
        
        Args:
            task_type: Type of task ('classification', 'regression', 'clustering', or 'time_series')
            X: Features DataFrame (optional, used for model selection)
            y: Target Series (optional, used for model selection)
            
        Returns:
            Dictionary of model configurations
        """
        from sklearn.ensemble import (
            GradientBoostingClassifier, GradientBoostingRegressor, 
            AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, 
            ExtraTreesRegressor, BaggingClassifier, BaggingRegressor
        )
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.naive_bayes import GaussianNB
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.linear_model import (
            Ridge, Lasso, ElasticNet, SGDClassifier, SGDRegressor,
            PassiveAggressiveClassifier, BayesianRidge, HuberRegressor
        )
        from sklearn.cluster import (
            KMeans, DBSCAN, AgglomerativeClustering, MeanShift,
            SpectralClustering, Birch, OPTICS
        )
        from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
        
        models = {}
        sample_size = len(X) if X is not None else 0
        n_features = X.shape[1] if X is not None and len(X.shape) > 1 else 0
        n_classes = len(np.unique(y)) if y is not None and task_type == 'classification' else 0
        
        # Common model parameters
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'random_state': [settings.RANDOM_STATE]
        }
        
        if task_type == 'classification':
            # ====== CLASSIFICATION MODELS ======
            models = {
                # Linear Models
                'logistic_regression': {
                    'model': LogisticRegression(max_iter=1000, random_state=settings.RANDOM_STATE),
                    'params': {
                        'model__C': [0.1, 1.0, 10.0],
                        'model__penalty': ['l2'],
                        'model__solver': ['lbfgs', 'saga']
                    },
                    'priority': 10
                },
                'sgd_classifier': {
                    'model': SGDClassifier(random_state=settings.RANDOM_STATE, max_iter=1000),
                    'params': {
                        'model__loss': ['hinge', 'log_loss'],
                        'model__penalty': ['l2', 'l1'],
                        'model__alpha': [0.0001, 0.001]
                    },
                    'priority': 7
                },
                'passive_aggressive': {
                    'model': PassiveAggressiveClassifier(random_state=settings.RANDOM_STATE, max_iter=1000),
                    'params': {
                        'model__C': [0.1, 1.0, 10.0]
                    },
                    'priority': 6
                },
                
                # Tree-Based Models
                'random_forest': {
                    'model': RandomForestClassifier(random_state=settings.RANDOM_STATE),
                    'params': {
                        'model__n_estimators': rf_params['n_estimators'],
                        'model__max_depth': rf_params['max_depth'],
                        'model__min_samples_split': rf_params['min_samples_split']
                    },
                    'priority': 10
                },
                'extra_trees': {
                    'model': ExtraTreesClassifier(random_state=settings.RANDOM_STATE),
                    'params': {
                        'model__n_estimators': [100, 200],
                        'model__max_depth': [10, 20, None]
                    },
                    'priority': 9
                },
                'decision_tree': {
                    'model': DecisionTreeClassifier(random_state=settings.RANDOM_STATE),
                    'params': {
                        'model__max_depth': [5, 10, 20, None],
                        'model__criterion': ['gini', 'entropy']
                    },
                    'priority': 7
                },
                
                # Boosting Models
                'gradient_boosting': {
                    'model': GradientBoostingClassifier(random_state=settings.RANDOM_STATE),
                    'params': {
                        'model__n_estimators': [100, 200],
                        'model__learning_rate': [0.01, 0.1],
                        'model__max_depth': [3, 5]
                    },
                    'priority': 10
                },
                'adaboost': {
                    'model': AdaBoostClassifier(random_state=settings.RANDOM_STATE),
                    'params': {
                        'model__n_estimators': [50, 100],
                        'model__learning_rate': [0.5, 1.0]
                    },
                    'priority': 9
                },
                'xgboost': {
                    'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', 
                                         random_state=settings.RANDOM_STATE),
                    'params': {
                        'model__n_estimators': [100, 200],
                        'model__learning_rate': [0.01, 0.1],
                        'model__max_depth': [3, 6]
                    },
                    'priority': 10
                },
                
                # Ensemble & Other Models
                'bagging': {
                    'model': BaggingClassifier(random_state=settings.RANDOM_STATE),
                    'params': {
                        'model__n_estimators': [10, 50],
                        'model__max_samples': [0.7, 1.0]
                    },
                    'priority': 8
                },
                'svc': {
                    'model': SVC(probability=True, random_state=settings.RANDOM_STATE),
                    'params': {
                        'model__C': [0.1, 1, 10],
                        'model__kernel': ['linear', 'rbf']
                    },
                    'priority': 8
                },
                'knn': {
                    'model': KNeighborsClassifier(),
                    'params': {
                        'model__n_neighbors': [3, 5, 7],
                        'model__weights': ['uniform', 'distance']
                    },
                    'priority': 7
                },
                'gaussian_nb': {
                    'model': GaussianNB(),
                    'params': {},
                    'priority': 7
                },
                'lda': {
                    'model': LinearDiscriminantAnalysis(),
                    'params': {},
                    'priority': 8
                },
                'qda': {
                    'model': QuadraticDiscriminantAnalysis(),
                    'params': {},
                    'priority': 7
                },
                'mlp': {
                    'model': MLPClassifier(random_state=settings.RANDOM_STATE, max_iter=500),
                    'params': {
                        'model__hidden_layer_sizes': [(50,), (100,)],
                        'model__activation': ['relu', 'tanh']
                    },
                    'priority': 8
                }
            }
            
            # Add LightGBM if available
            if LIGHTGBM_AVAILABLE:
                models['lightgbm'] = {
                    'model': lgb.LGBMClassifier(random_state=settings.RANDOM_STATE, verbose=-1),
                    'params': {
                        'model__n_estimators': [100, 200],
                        'model__learning_rate': [0.05, 0.1],
                        'model__num_leaves': [31, 50]
                    },
                    'priority': 10
                }
        
        elif task_type == 'regression':
            # ====== REGRESSION MODELS ======
            models = {
                # Linear Models
                'linear_regression': {
                    'model': LinearRegression(),
                    'params': {},
                    'priority': 10
                },
                'ridge': {
                    'model': Ridge(random_state=settings.RANDOM_STATE),
                    'params': {
                        'model__alpha': [0.1, 1.0, 10.0, 100.0]
                    },
                    'priority': 10
                },
                'lasso': {
                    'model': Lasso(random_state=settings.RANDOM_STATE, max_iter=2000),
                    'params': {
                        'model__alpha': [0.1, 1.0, 10.0]
                    },
                    'priority': 10
                },
                'elasticnet': {
                    'model': ElasticNet(random_state=settings.RANDOM_STATE, max_iter=2000),
                    'params': {
                        'model__alpha': [0.1, 1.0],
                        'model__l1_ratio': [0.2, 0.5, 0.8]
                    },
                    'priority': 9
                },
                'bayesian_ridge': {
                    'model': BayesianRidge(),
                    'params': {},
                    'priority': 8
                },
                'huber': {
                    'model': HuberRegressor(max_iter=200),
                    'params': {
                        'model__epsilon': [1.1, 1.35, 1.5]
                    },
                    'priority': 8
                },
                'sgd_regressor': {
                    'model': SGDRegressor(random_state=settings.RANDOM_STATE, max_iter=1000),
                    'params': {
                        'model__loss': ['squared_error', 'huber'],
                        'model__penalty': ['l2', 'l1']
                    },
                    'priority': 7
                },
                
                # Tree-Based Models
                'random_forest': {
                    'model': RandomForestRegressor(random_state=settings.RANDOM_STATE),
                    'params': {
                        'model__n_estimators': rf_params['n_estimators'],
                        'model__max_depth': rf_params['max_depth'],
                        'model__min_samples_split': rf_params['min_samples_split']
                    },
                    'priority': 10
                },
                'extra_trees': {
                    'model': ExtraTreesRegressor(random_state=settings.RANDOM_STATE),
                    'params': {
                        'model__n_estimators': [100, 200],
                        'model__max_depth': [10, 20, None]
                    },
                    'priority': 9
                },
                'decision_tree': {
                    'model': DecisionTreeRegressor(random_state=settings.RANDOM_STATE),
                    'params': {
                        'model__max_depth': [5, 10, 20, None]
                    },
                    'priority': 7
                },
                
                # Boosting Models
                'gradient_boosting': {
                    'model': GradientBoostingRegressor(random_state=settings.RANDOM_STATE),
                    'params': {
                        'model__n_estimators': [100, 200],
                        'model__learning_rate': [0.01, 0.1],
                        'model__max_depth': [3, 5]
                    },
                    'priority': 10
                },
                'adaboost': {
                    'model': AdaBoostRegressor(random_state=settings.RANDOM_STATE),
                    'params': {
                        'model__n_estimators': [50, 100],
                        'model__learning_rate': [0.5, 1.0]
                    },
                    'priority': 9
                },
                'xgboost': {
                    'model': XGBRegressor(random_state=settings.RANDOM_STATE),
                    'params': {
                        'model__n_estimators': [100, 200],
                        'model__learning_rate': [0.01, 0.1],
                        'model__max_depth': [3, 6]
                    },
                    'priority': 10
                },
                
                # Ensemble & Other Models
                'bagging': {
                    'model': BaggingRegressor(random_state=settings.RANDOM_STATE),
                    'params': {
                        'model__n_estimators': [10, 50]
                    },
                    'priority': 8
                },
                'svr': {
                    'model': SVR(),
                    'params': {
                        'model__C': [0.1, 1, 10],
                        'model__kernel': ['linear', 'rbf']
                    },
                    'priority': 7
                },
                'knn': {
                    'model': KNeighborsRegressor(),
                    'params': {
                        'model__n_neighbors': [3, 5, 7],
                        'model__weights': ['uniform', 'distance']
                    },
                    'priority': 7
                },
                'mlp': {
                    'model': MLPRegressor(random_state=settings.RANDOM_STATE, max_iter=500),
                    'params': {
                        'model__hidden_layer_sizes': [(50,), (100,)],
                        'model__activation': ['relu', 'tanh']
                    },
                    'priority': 8
                }
            }
            
            # Add LightGBM if available
            if LIGHTGBM_AVAILABLE:
                models['lightgbm'] = {
                    'model': lgb.LGBMRegressor(random_state=settings.RANDOM_STATE, verbose=-1),
                    'params': {
                        'model__n_estimators': [100, 200],
                        'model__learning_rate': [0.05, 0.1],
                        'model__num_leaves': [31, 50]
                    },
                    'priority': 10
                }
        
        elif task_type == 'clustering':
            # ====== CLUSTERING MODELS ======
            models = {
                'kmeans': {
                    'model': KMeans(random_state=settings.RANDOM_STATE, n_init=10),
                    'params': {
                        'n_clusters': [2, 3, 4, 5, 6, 7, 8]
                    },
                    'priority': 10
                },
                'gaussian_mixture': {
                    'model': GaussianMixture(random_state=settings.RANDOM_STATE),
                    'params': {
                        'n_components': [2, 3, 4, 5, 6],
                        'covariance_type': ['full', 'tied', 'diag']
                    },
                    'priority': 9
                },
                'bayesian_gaussian_mixture': {
                    'model': BayesianGaussianMixture(random_state=settings.RANDOM_STATE),
                    'params': {
                        'n_components': [2, 3, 4, 5, 6]
                    },
                    'priority': 8
                },
                'dbscan': {
                    'model': DBSCAN(),
                    'params': {
                        'eps': [0.3, 0.5, 0.7, 1.0],
                        'min_samples': [3, 5, 10]
                    },
                    'priority': 9
                },
                'agglomerative': {
                    'model': AgglomerativeClustering(),
                    'params': {
                        'n_clusters': [2, 3, 4, 5, 6, 7, 8],
                        'linkage': ['ward', 'complete', 'average']
                    },
                    'priority': 9
                },
                'spectral': {
                    'model': SpectralClustering(random_state=settings.RANDOM_STATE),
                    'params': {
                        'n_clusters': [2, 3, 4, 5, 6]
                    },
                    'priority': 7
                },
                'birch': {
                    'model': Birch(),
                    'params': {
                        'n_clusters': [2, 3, 4, 5, 6, 7, 8]
                    },
                    'priority': 8
                },
                'optics': {
                    'model': OPTICS(),
                    'params': {
                        'min_samples': [3, 5, 10]
                    },
                    'priority': 7
                },
                'meanshift': {
                    'model': MeanShift(),
                    'params': {},
                    'priority': 6
                }
            }
        
        elif task_type == 'time_series':
            # ====== TIME SERIES MODELS ======
            models = {
                # Classical Time Series Models
                'arima': {
                    'model': 'ARIMA',
                    'params': {
                        'order': [(1,1,1), (2,1,1), (1,1,2), (2,1,2), (3,1,1)]
                    },
                    'priority': 10
                },
                'sarima': {
                    'model': 'SARIMAX',
                    'params': {
                        'order': [(1,1,1), (2,1,1)],
                        'seasonal_order': [(1,1,1,12), (0,1,1,12), (1,0,1,12)]
                    },
                    'priority': 10
                },
                'auto_arima': {
                    'model': 'AutoARIMA',
                    'params': {},
                    'priority': 10
                },
                'exponential_smoothing': {
                    'model': 'ExponentialSmoothing',
                    'params': {
                        'seasonal': ['add', 'mul', None],
                        'trend': ['add', 'mul', None]
                    },
                    'priority': 10
                },
                'holt_winters': {
                    'model': 'HoltWinters',
                    'params': {
                        'seasonal': ['add', 'mul'],
                        'seasonal_periods': [12, 7, 4]
                    },
                    'priority': 9
                },
                'simple_exponential_smoothing': {
                    'model': 'SimpleExpSmoothing',
                    'params': {
                        'smoothing_level': [0.2, 0.5, 0.8]
                    },
                    'priority': 8
                },
                
                # Advanced Time Series Models
                'prophet': {
                    'model': 'Prophet',
                    'params': {
                        'changepoint_prior_scale': [0.05, 0.5],
                        'seasonality_prior_scale': [1.0, 10.0]
                    },
                    'priority': 10
                },
                'var': {
                    'model': 'VAR',
                    'params': {
                        'maxlags': [5, 10, 15]
                    },
                    'priority': 8
                },
                'vecm': {
                    'model': 'VECM',
                    'params': {
                        'k_ar_diff': [1, 2, 3]
                    },
                    'priority': 7
                },
                
                # State Space Models
                'unobserved_components': {
                    'model': 'UnobservedComponents',
                    'params': {
                        'level': ['local level', 'local linear trend'],
                        'seasonal': [None, 12]
                    },
                    'priority': 8
                },
                'dynamic_factor': {
                    'model': 'DynamicFactor',
                    'params': {
                        'k_factors': [1, 2],
                        'factor_order': [1, 2]
                    },
                    'priority': 7
                },
                
                # Machine Learning for Time Series
                'lstm': {
                    'model': 'LSTM',
                    'params': {
                        'units': [50, 100],
                        'epochs': [50, 100],
                        'batch_size': [16, 32]
                    },
                    'priority': 9
                },
                'gru': {
                    'model': 'GRU',
                    'params': {
                        'units': [50, 100],
                        'epochs': [50, 100],
                        'batch_size': [16, 32]
                    },
                    'priority': 9
                },
                'tcn': {
                    'model': 'TCN',
                    'params': {
                        'nb_filters': [32, 64],
                        'kernel_size': [2, 3],
                        'dilations': [[1, 2, 4, 8]]
                    },
                    'priority': 8
                },
                
                # Tree-Based for Time Series
                'random_forest_ts': {
                    'model': RandomForestRegressor(random_state=settings.RANDOM_STATE),
                    'params': {
                        'model__n_estimators': [100, 200],
                        'model__max_depth': [10, 20, None]
                    },
                    'priority': 8
                },
                'xgboost_ts': {
                    'model': XGBRegressor(random_state=settings.RANDOM_STATE),
                    'params': {
                        'model__n_estimators': [100, 200],
                        'model__learning_rate': [0.01, 0.1],
                        'model__max_depth': [3, 6]
                    },
                    'priority': 9
                },
                'lightgbm_ts': {
                    'model': lgb.LGBMRegressor(random_state=settings.RANDOM_STATE, verbose=-1) if LIGHTGBM_AVAILABLE else None,
                    'params': {
                        'model__n_estimators': [100, 200],
                        'model__learning_rate': [0.05, 0.1]
                    },
                    'priority': 9
                },
                
                # Theta Method
                'theta': {
                    'model': 'Theta',
                    'params': {
                        'theta': [0, 1, 2]
                    },
                    'priority': 8
                },
                
                # TBATS
                'tbats': {
                    'model': 'TBATS',
                    'params': {},
                    'priority': 8
                },
                
                # Croston for Intermittent Demand
                'croston': {
                    'model': 'Croston',
                    'params': {
                        'alpha': [0.1, 0.2, 0.3]
                    },
                    'priority': 7
                }
            }
            
            # Remove None models (e.g., if LightGBM not available)
            models = {k: v for k, v in models.items() if v['model'] is not None}
        
        # Sort models by priority and select top models based on task type
        sorted_models = sorted(
            models.items(), 
            key=lambda x: x[1].get('priority', 0),
            reverse=True
        )
        
        # Select appropriate number of models based on task type and dataset size
        if task_type in ['clustering', 'time_series']:
            # For clustering and time series, use all available models
            max_models = len(sorted_models)
        else:
            # For classification/regression, select based on dataset size
            if sample_size > 5000:
                max_models = min(10, len(sorted_models))  # More models for larger datasets
            elif sample_size > 1000:
                max_models = min(8, len(sorted_models))
            else:
                max_models = min(6, len(sorted_models))  # Fewer models for small datasets
            
        selected_models = dict(sorted_models[:max_models])
        
        # Remove priority key before returning
        for model_cfg in selected_models.values():
            model_cfg.pop('priority', None)
            
        return selected_models
    
    async def _train_single_model(self, model_name: str, model_config: Dict[str, Any],
                                X_train: pd.DataFrame, X_test: pd.DataFrame,
                                y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """
        Train a single model with hyperparameter optimization
        
        Handles special cases for time series and clustering models
        """
        
        # Special handling for time series models
        if self.task_type == 'time_series':
            return await self._train_time_series_model(model_name, model_config, X_train, X_test, y_train, y_test)
            
        # Special handling for clustering models
        if self.task_type == 'clustering':
            return await self._train_clustering_model(model_name, model_config, X_train, X_test)
            
        # For supervised learning (classification/regression)
        # Create pipeline
        pipeline_steps = []
        
        # Add preprocessor if it exists and we have features to preprocess
        if self.preprocessor is not None and X_train is not None and len(X_train) > 0:
            pipeline_steps.append(('preprocessor', self.preprocessor))
            
        # Add the model
        pipeline_steps.append(('model', model_config['model']))
        
        pipeline = Pipeline(pipeline_steps)
        
        # Hyperparameter optimization
        if model_config['params'] and OPTUNA_AVAILABLE:
            best_params = await self._optimize_hyperparameters(
                pipeline, model_config['params'], X_train, y_train
            )
            
            # Update model with best parameters
            for param, value in best_params.items():
                if param.startswith('model__'):
                    param_name = param.replace('model__', '')
                    setattr(pipeline.named_steps['model'], param_name, value)
        
        # Train final model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
        
        # Calculate metrics
        metrics = await self._calculate_metrics(
            y_train, 
            y_pred_train if hasattr(pipeline, 'predict') else None,
            y_test,
            y_pred_test if hasattr(pipeline, 'predict') else None,
            pipeline, 
            X_test
        )
        
        # Save model
        model_id = str(uuid.uuid4())
        model_path = settings.MODELS_DIR / f"{model_name}_{model_id}.joblib"
        
        try:
            # Special handling for models that can't be pickled directly
            if hasattr(pipeline.named_steps.get('model', pipeline), 'save'):
                # For models with save method (like Keras)
                pipeline.named_steps['model'].save(str(model_path))
            else:
                joblib.dump(pipeline, model_path)
                
            return {
                "model_name": model_name,
                "model_id": model_id,
                "model_path": str(model_path),
                "metrics": metrics,
                "status": "completed",
                "trained_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return {
                "model_name": model_name,
                "status": "failed",
                "error": f"Model training completed but saving failed: {str(e)}"
            }
    
    async def _train_time_series_model(self, model_name: str, model_config: Dict[str, Any],
                                     X_train: pd.DataFrame, X_test: pd.DataFrame,
                                     y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """Special handling for time series models"""
        try:
            if model_name == 'prophet':
                # Prophet requires specific data format
                from prophet import Prophet
                
                # Prepare data for Prophet
                train_df = pd.DataFrame({
                    'ds': X_train.index if hasattr(X_train, 'index') else range(len(X_train)),
                    'y': y_train.values
                })
                
                # Create and fit model
                model = Prophet(
                    seasonality_mode=model_config['params'].get('model__seasonality_mode', ['additive'])[0],
                    changepoint_prior_scale=model_config['params'].get('model__changepoint_prior_scale', [0.05])[0]
                )
                model.fit(train_df)
                
                # Make predictions
                future = model.make_future_dataframe(periods=len(y_test))
                forecast = model.predict(future)
                y_pred = forecast['yhat'].values[-len(y_test):]
                
                # Calculate metrics
                metrics = {
                    'test': {
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'mae': mean_absolute_error(y_test, y_pred),
                        'r2': r2_score(y_test, y_pred)
                    },
                    'primary_metric': 'rmse'
                }
                
                # Save model
                model_id = str(uuid.uuid4())
                model_path = settings.MODELS_DIR / f"prophet_{model_id}.json"
                
                # Save model components for later use
                model_components = {
                    'model': model,
                    'last_training_date': X_train.index[-1] if hasattr(X_train, 'index') else len(X_train)
                }
                joblib.dump(model_components, model_path)
                
                return {
                    'model_name': 'prophet',
                    'model_id': model_id,
                    'model_path': str(model_path),
                    'metrics': metrics,
                    'status': 'completed',
                    'trained_at': datetime.utcnow().isoformat()
                }
                
            else:  # ARIMA, ExponentialSmoothing, etc.
                # Fall back to standard training for other time series models
                return await self._train_standard_model(model_name, model_config, X_train, X_test, y_train, y_test)
                
        except Exception as e:
            logger.error(f"Time series model training failed: {str(e)}")
            return {
                'model_name': model_name,
                'status': 'failed',
                'error': str(e)
            }
    
    async def _train_clustering_model(self, model_name: str, model_config: Dict[str, Any],
                                    X_train: pd.DataFrame, X_test: pd.DataFrame) -> Dict[str, Any]:
        """Special handling for clustering models"""
        try:
            # Create and fit the model
            model = model_config['model']
            
            # Handle different clustering algorithms
            if hasattr(model, 'fit_predict'):
                labels = model.fit_predict(X_train)
            else:
                model.fit(X_train)
                labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X_train)
            
            # Calculate metrics
            metrics = await self._calculate_metrics(None, None, None, None, model, X_train)
            
            # Save model
            model_id = str(uuid.uuid4())
            model_path = settings.MODELS_DIR / f"{model_name}_{model_id}.joblib"
            joblib.dump(model, model_path)
            
            return {
                'model_name': model_name,
                'model_id': model_id,
                'model_path': str(model_path),
                'metrics': metrics,
                'status': 'completed',
                'trained_at': datetime.utcnow().isoformat(),
                'cluster_labels': labels.tolist() if hasattr(labels, 'tolist') else labels
            }
            
        except Exception as e:
            logger.error(f"Clustering model training failed: {str(e)}")
            return {
                'model_name': model_name,
                'status': 'failed',
                'error': str(e)
            }
    
    async def _train_standard_model(self, model_name: str, model_config: Dict[str, Any],
                                  X_train: pd.DataFrame, X_test: pd.DataFrame,
                                  y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """Standard model training for supervised learning"""
        # Create pipeline
        pipeline_steps = []
        
        # Add preprocessor if it exists and we have features to preprocess
        if self.preprocessor is not None and X_train is not None and len(X_train) > 0:
            pipeline_steps.append(('preprocessor', self.preprocessor))
            
        # Add the model
        pipeline_steps.append(('model', model_config['model']))
        
        pipeline = Pipeline(pipeline_steps)
        
        # Hyperparameter optimization for supervised learning
        if model_config.get('params') and OPTUNA_AVAILABLE and self.task_type in ['classification', 'regression']:
            try:
                best_params = await self._optimize_hyperparameters(
                    pipeline, model_config['params'], X_train, y_train
                )
                
                # Update model with best parameters
                for param, value in best_params.items():
                    if param.startswith('model__'):
                        param_name = param.replace('model__', '')
                        if hasattr(pipeline.named_steps['model'], param_name):
                            setattr(pipeline.named_steps['model'], param_name, value)
            except Exception as e:
                logger.warning(f"Hyperparameter optimization failed: {str(e)}")
        
        # Train final model
        try:
            if y_train is not None and len(y_train) > 0:
                pipeline.fit(X_train, y_train)
            else:
                pipeline.fit(X_train)
                
            # Make predictions
            y_pred_train = pipeline.predict(X_train) if X_train is not None and len(X_train) > 0 else None
            y_pred_test = pipeline.predict(X_test) if X_test is not None and len(X_test) > 0 else None
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return {
                "model_name": model_name,
                "status": "failed",
                "error": str(e)
            }
            
        # Calculate metrics
        metrics = await self._calculate_metrics(
            y_train, 
            y_pred_train if hasattr(pipeline, 'predict') else None,
            y_test,
            y_pred_test if hasattr(pipeline, 'predict') else None,
            pipeline, 
            X_test
        )
        
        # Save model
        model_id = str(uuid.uuid4())
        model_path = settings.MODELS_DIR / f"{model_name}_{model_id}.joblib"
        
        try:
            joblib.dump(pipeline, model_path)
            
            return {
                "model_name": model_name,
                "model_id": model_id,
                "model_path": str(model_path),
                "metrics": metrics,
                "status": "completed",
                "trained_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return {
                "model_name": model_name,
                "status": "failed",
                "error": f"Model training completed but saving failed: {str(e)}"
            }
    
    async def _optimize_hyperparameters(self, pipeline: Pipeline, param_grid: Dict[str, List],
                                      X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna with task-specific optimization"""
        
        def objective(trial):
            # Sample parameters
            params = {}
            for param, values in param_grid.items():
                if not values:  # Skip empty parameter lists
                    continue
                    
                param_name = param.replace('model__', '')
                
                # Skip parameters not applicable to this model
                if not hasattr(pipeline.named_steps['model'], param_name):
                    continue
                    
                if isinstance(values[0], bool):
                    params[param] = trial.suggest_categorical(param, values)
                elif all(isinstance(x, int) for x in values):
                    params[param] = trial.suggest_int(param, min(values), max(values))
                elif all(isinstance(x, float) for x in values):
                    params[param] = trial.suggest_float(param, min(values), max(values), log=any(x > 0 and x < 1 for x in values))
                else:
                    params[param] = trial.suggest_categorical(param, values)
            
            # Skip if no valid parameters were found
            if not params:
                return float('-inf') if self.task_type == 'classification' else float('inf')
            
            try:
                # Create a new model instance with sampled parameters
                model_params = {k.replace('model__', ''): v for k, v in params.items()}
                model = pipeline.named_steps['model'].__class__(**model_params)
                
                # Create a new pipeline with the model
                temp_pipeline = Pipeline([
                    (name, step) for name, step in pipeline.steps 
                    if name != 'model'
                ] + [('model', model)])
                
                # Cross-validation with appropriate strategy
                if self.task_type == 'classification':
                    cv = StratifiedKFold(n_splits=min(5, len(np.unique(y_train))), 
                                       shuffle=True, 
                                       random_state=settings.RANDOM_STATE)
                    scoring = 'accuracy'
                else:  # regression
                    cv = KFold(n_splits=5, shuffle=True, random_state=settings.RANDOM_STATE)
                    scoring = 'neg_mean_squared_error'
                
                # Calculate cross-validated score
                scores = cross_val_score(
                    temp_pipeline, X_train, y_train, 
                    cv=cv, scoring=scoring, n_jobs=-1
                )
                
                return np.mean(scores)
                
            except Exception as e:
                logger.warning(f"Trial failed: {str(e)}")
                return float('-inf') if self.task_type == 'classification' else float('inf')
        
        # Run optimization
        try:
            direction = 'maximize' if self.task_type == 'classification' else 'minimize'
            study = optuna.create_study(direction=direction)
            
            # Adjust number of trials based on parameter space size
            n_trials = min(50, max(10, len(param_grid) * 5))
            study.optimize(objective, n_trials=n_trials, timeout=300)  # 5 minutes max
            
            return study.best_params
            
        except Exception as e:
            logger.warning(f"Hyperparameter optimization failed: {str(e)}")
            return {}
    
    async def _calculate_metrics(self, y_train: pd.Series, y_pred_train: np.ndarray,
                               y_test: pd.Series, y_pred_test: np.ndarray,
                               pipeline: Pipeline, X_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for model evaluation
        
        Args:
            y_train: Training target values
            y_pred_train: Training predictions
            y_test: Test target values
            y_pred_test: Test predictions
            pipeline: Trained model pipeline
            X_test: Test features
            
        Returns:
            Dictionary of calculated metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
            mean_squared_error, mean_absolute_error, r2_score, log_loss, 
            mean_absolute_percentage_error, confusion_matrix, classification_report,
            mean_squared_log_error, max_error, explained_variance_score, 
            balanced_accuracy_score, top_k_accuracy_score, silhouette_score,
            calinski_harabasz_score, davies_bouldin_score
        )
        
        metrics = {}
        
        if self.task_type == 'clustering':
            # Clustering metrics
            try:
                labels = pipeline.predict(X_test) if hasattr(pipeline, 'predict') else pipeline.labels_
                
                # Only calculate if we have valid clusters (more than 1 and less than n_samples)
                n_clusters = len(np.unique(labels))
                if 1 < n_clusters < len(X_test):
                    metrics['test'] = {
                        'silhouette_score': float(silhouette_score(X_test, labels)),
                        'calinski_harabasz_score': float(calinski_harabasz_score(X_test, labels)),
                        'davies_bouldin_score': float(davies_bouldin_score(X_test, labels)),
                        'n_clusters': int(n_clusters)
                    }
                    metrics['primary_metric'] = 'silhouette_score'
                else:
                    metrics['test'] = {
                        'n_clusters': int(n_clusters),
                        'warning': 'Too few or too many clusters for meaningful metrics'
                    }
                    metrics['primary_metric'] = 'n_clusters'
                    
            except Exception as e:
                logger.warning(f"Error calculating clustering metrics: {str(e)}")
                metrics['test'] = {'error': str(e)}
                metrics['primary_metric'] = None
                
        elif self.task_type == 'time_series':
            # Time series specific metrics
            try:
                # Calculate time series specific metrics
                residuals = y_test - y_pred_test
                
                metrics['test'] = {
                    'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
                    'mae': float(mean_absolute_error(y_test, y_pred_test)),
                    'mape': float(mean_absolute_percentage_error(
                        y_test, 
                        np.clip(y_pred_test, 1e-10, None)  # Avoid division by zero
                    )) if (y_test != 0).any() else None,
                    'r2': float(r2_score(y_test, y_pred_test)),
                    'mean_absolute_scaled_error': float(
                        np.mean(np.abs(residuals)) / np.mean(np.abs(np.diff(y_test)))
                    ) if len(y_test) > 1 else None
                }
                
                # Remove None values
                metrics['test'] = {k: v for k, v in metrics['test'].items() if v is not None}
                metrics['primary_metric'] = 'rmse'
                
                # Add residual analysis
                metrics['residuals'] = {
                    'mean': float(np.mean(residuals)),
                    'std': float(np.std(residuals)),
                    'autocorrelation': float(pd.Series(residuals).autocorr())
                }
                
            except Exception as e:
                logger.warning(f"Error calculating time series metrics: {str(e)}")
                metrics['test'] = {'error': str(e)}
                metrics['primary_metric'] = None
                
        elif self.task_type == 'classification':
            n_classes = len(np.unique(y_test))
            is_binary = n_classes == 2
            
            # Common classification metrics
            common_metrics = {
                'accuracy': accuracy_score(y_test, y_pred_test),
                'precision_macro': precision_score(y_test, y_pred_test, average='macro', zero_division=0),
                'precision_weighted': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'recall_macro': recall_score(y_test, y_pred_test, average='macro', zero_division=0),
                'recall_weighted': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'f1_macro': f1_score(y_test, y_pred_test, average='macro', zero_division=0),
                'f1_weighted': f1_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_test)
            }
            
            # Binary classification specific metrics
            if is_binary:
                try:
                    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                    common_metrics.update({
                        'roc_auc': roc_auc_score(y_test, y_pred_proba),
                        'log_loss': log_loss(y_test, y_pred_proba, labels=np.unique(y_test))
                    })
                except Exception as e:
                    logger.warning(f"Could not calculate probability metrics: {str(e)}")
            
            # Multi-class specific metrics
            if n_classes > 2:
                common_metrics.update({
                    'top_k_accuracy': top_k_accuracy_score(
                        y_test, pipeline.predict_proba(X_test), k=min(3, n_classes)
                    ) if n_classes > 2 else None
                })
            
            # Training metrics
            metrics['train'] = {
                'accuracy': float(accuracy_score(y_train, y_pred_train)),
                'f1_weighted': float(f1_score(y_train, y_pred_train, average='weighted', zero_division=0)),
                'log_loss': float(log_loss(y_train, pipeline.predict_proba(X_test), labels=np.unique(y_train)))
            }
            
            # Test metrics
            metrics['test'] = {k: float(v) for k, v in common_metrics.items() if v is not None}
            
            # Confusion matrix (only for small number of classes)
            if n_classes <= 10:
                cm = confusion_matrix(y_test, y_pred_test)
                metrics['confusion_matrix'] = cm.tolist()
                metrics['class_report'] = classification_report(
                    y_test, y_pred_test, output_dict=True, zero_division=0
                )
            
            metrics['primary_metric'] = 'roc_auc' if is_binary else 'f1_weighted'
            
        else:  # regression
            # Calculate all possible metrics
            all_metrics = {
                'mse': mean_squared_error(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'mae': mean_absolute_error(y_test, y_pred_test),
                'r2': r2_score(y_test, y_pred_test),
                'explained_variance': explained_variance_score(y_test, y_pred_test),
                'max_error': max_error(y_test, y_pred_test),
                'msle': mean_squared_log_error(
                    np.maximum(0, y_test),  # Ensure non-negative for log
                    np.maximum(0, y_pred_test)
                ) if (y_test >= 0).all() and (y_pred_test >= 0).all() else None,
                'mape': mean_absolute_percentage_error(
                    y_test, 
                    np.clip(y_pred_test, 1e-10, None)  # Avoid division by zero
                ) if (y_test != 0).any() else None
            }
            
            # Training metrics (simplified)
            metrics['train'] = {
                'rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
                'mae': float(mean_absolute_error(y_train, y_pred_train)),
                'r2': float(r2_score(y_train, y_pred_train))
            }
            
            # Test metrics (all available metrics)
            metrics['test'] = {k: float(v) for k, v in all_metrics.items() if v is not None}
            
            # Residual analysis
            residuals = y_test - y_pred_test
            metrics['residuals'] = {
                'mean': float(np.mean(residuals)),
                'std': float(np.std(residuals)),
                'min': float(np.min(residuals)),
                '25%': float(np.percentile(residuals, 25)),
                '50%': float(np.median(residuals)),
                '75%': float(np.percentile(residuals, 75)),
                'max': float(np.max(residuals))
            }
            
            metrics['primary_metric'] = 'rmse'
        
        return metrics
    
    async def _generate_intelligent_insights(self, model_name: str, test_metrics: Dict, 
                                            primary_metric: str, primary_value: float,
                                            base_characteristics: Dict) -> tuple:
        """
        Generate intelligent pros/cons based on actual model performance and dataset characteristics
        """
        intelligent_pros = list(base_characteristics['pros'][:2])  # Start with top 2 standard pros
        intelligent_cons = list(base_characteristics['cons'][:1])  # Start with top 1 standard con
        
        # Analyze performance metrics for intelligent insights
        if self.task_type == 'classification':
            accuracy = test_metrics.get('accuracy', 0)
            f1 = test_metrics.get('f1_weighted', test_metrics.get('f1', 0))
            precision = test_metrics.get('precision_weighted', test_metrics.get('precision', 0))
            recall = test_metrics.get('recall_weighted', test_metrics.get('recall', 0))
            
            # Performance-based pros
            if accuracy > 0.9:
                intelligent_pros.append(f"Excellent accuracy ({accuracy:.2%}) on this dataset")
            elif accuracy > 0.8:
                intelligent_pros.append(f"Strong accuracy ({accuracy:.2%}) for this task")
            
            if f1 > 0.85:
                intelligent_pros.append(f"Balanced precision-recall (F1: {f1:.3f})")
            
            if precision > recall + 0.1:
                intelligent_pros.append(f"High precision ({precision:.3f}) - fewer false positives")
            elif recall > precision + 0.1:
                intelligent_pros.append(f"High recall ({recall:.3f}) - catches most positives")
            
            # Performance-based cons
            if accuracy < 0.7:
                intelligent_cons.append(f"Lower accuracy ({accuracy:.2%}) on this dataset")
            if abs(precision - recall) > 0.15:
                intelligent_cons.append(f"Imbalanced precision-recall trade-off")
                
        elif self.task_type == 'regression':
            r2 = test_metrics.get('r2', 0)
            rmse = test_metrics.get('rmse', 0)
            mae = test_metrics.get('mae', 0)
            mape = test_metrics.get('mape', 0)
            
            # Performance-based pros
            if r2 > 0.9:
                intelligent_pros.append(f"Excellent fit (R²={r2:.3f}) explains variance well")
            elif r2 > 0.7:
                intelligent_pros.append(f"Good fit (R²={r2:.3f}) captures patterns effectively")
            
            if mape and mape < 10:
                intelligent_pros.append(f"Low error rate (MAPE: {mape:.1f}%)")
            
            if rmse < mae * 1.2:
                intelligent_pros.append("Consistent predictions with few large errors")
            
            # Performance-based cons
            if r2 < 0.5:
                intelligent_cons.append(f"Moderate fit (R²={r2:.3f}) - may miss some patterns")
            if mape and mape > 20:
                intelligent_cons.append(f"Higher error rate (MAPE: {mape:.1f}%)")
            if rmse > mae * 1.5:
                intelligent_cons.append("Some large prediction errors present")
                
        elif self.task_type == 'clustering':
            silhouette = test_metrics.get('silhouette_score', 0)
            davies_bouldin = test_metrics.get('davies_bouldin_score', float('inf'))
            n_clusters = test_metrics.get('n_clusters', 0)
            
            # Performance-based pros
            if silhouette > 0.5:
                intelligent_pros.append(f"Well-separated clusters (Silhouette: {silhouette:.3f})")
            elif silhouette > 0.3:
                intelligent_pros.append(f"Decent cluster separation (Silhouette: {silhouette:.3f})")
            
            if davies_bouldin < 1.0:
                intelligent_pros.append(f"Compact clusters (Davies-Bouldin: {davies_bouldin:.3f})")
            
            if 2 <= n_clusters <= 10:
                intelligent_pros.append(f"Optimal cluster count ({n_clusters}) for interpretation")
            
            # Performance-based cons
            if silhouette < 0.2:
                intelligent_cons.append(f"Weak cluster separation (Silhouette: {silhouette:.3f})")
            if davies_bouldin > 2.0:
                intelligent_cons.append(f"Overlapping clusters (Davies-Bouldin: {davies_bouldin:.3f})")
                
        elif self.task_type == 'time_series':
            rmse = test_metrics.get('rmse', 0)
            mape = test_metrics.get('mape', 0)
            r2 = test_metrics.get('r2', 0)
            
            # Performance-based pros
            if r2 > 0.8:
                intelligent_pros.append(f"Captures temporal patterns well (R²={r2:.3f})")
            if mape and mape < 15:
                intelligent_pros.append(f"Accurate forecasts (MAPE: {mape:.1f}%)")
            
            # Performance-based cons
            if r2 < 0.6:
                intelligent_cons.append(f"Misses some temporal patterns (R²={r2:.3f})")
            if mape and mape > 25:
                intelligent_cons.append(f"Higher forecast errors (MAPE: {mape:.1f}%)")
        
        # Dataset-specific insights
        if hasattr(self, 'df') and self.df is not None:
            n_samples = len(self.df)
            n_features = self.df.shape[1] if len(self.df.shape) > 1 else 1
            
            # Add dataset-specific pros
            if n_samples < 1000 and 'simple' in model_name.lower():
                intelligent_pros.append(f"Well-suited for small dataset ({n_samples} samples)")
            elif n_samples > 10000 and 'xgb' in model_name.lower() or 'lightgbm' in model_name.lower():
                intelligent_pros.append(f"Scales well with large dataset ({n_samples:,} samples)")
            
            if n_features > 50 and ('forest' in model_name.lower() or 'xgb' in model_name.lower()):
                intelligent_pros.append(f"Handles high-dimensional data ({n_features} features)")
            
            # Add dataset-specific cons
            if n_samples < 500 and ('mlp' in model_name.lower() or 'deep' in model_name.lower()):
                intelligent_cons.append(f"Limited data ({n_samples} samples) for complex model")
            elif n_samples > 50000 and 'svc' in model_name.lower():
                intelligent_cons.append(f"Slow on large dataset ({n_samples:,} samples)")
        
        # Try to use Gemini for enhanced insights (optional, with fallback)
        try:
            gemini_insights = await self._get_gemini_insights(model_name, test_metrics, self.task_type)
            if gemini_insights:
                # Add top Gemini insight to pros if available
                if gemini_insights.get('additional_pro'):
                    intelligent_pros.append(gemini_insights['additional_pro'])
                if gemini_insights.get('additional_con'):
                    intelligent_cons.append(gemini_insights['additional_con'])
        except Exception as e:
            logger.debug(f"Gemini insights not available: {str(e)}")
        
        # Limit to top 4 pros and top 3 cons
        return intelligent_pros[:4], intelligent_cons[:3]
    
    async def _get_gemini_insights(self, model_name: str, metrics: Dict, task_type: str) -> Dict:
        """
        Get AI-generated insights from Gemini about model performance
        """
        try:
            import google.generativeai as genai
            import os
            
            # Check if Gemini API key is available
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                return {}
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            # Create a concise prompt
            metrics_str = ', '.join([f"{k}: {v}" for k, v in metrics.items() if v != 'N/A'])
            prompt = f"""Analyze this {task_type} model performance:
Model: {model_name}
Metrics: {metrics_str}

Provide ONE additional specific advantage and ONE additional specific concern about this model's performance on this dataset. Be concise (max 10 words each).

Format:
Advantage: [your insight]
Concern: [your insight]"""

            response = model.generate_content(prompt)
            
            if response and response.text:
                lines = response.text.strip().split('\n')
                insights = {}
                for line in lines:
                    if 'advantage:' in line.lower():
                        insights['additional_pro'] = line.split(':', 1)[1].strip()
                    elif 'concern:' in line.lower():
                        insights['additional_con'] = line.split(':', 1)[1].strip()
                return insights
                
        except Exception as e:
            logger.debug(f"Gemini API call failed: {str(e)}")
            return {}
        
        return {}
    
    async def _create_model_comparison(self, models_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create comprehensive model comparison table with rankings, advantages, and disadvantages
        Returns top 6-8 models with detailed insights
        """
        
        # Model characteristics database
        model_characteristics = {
            # Classification
            'logistic_regression': {
                'pros': ['Fast training', 'Interpretable coefficients', 'Works well with linear relationships', 'Low memory footprint'],
                'cons': ['Assumes linear decision boundary', 'May underfit complex patterns', 'Sensitive to feature scaling']
            },
            'random_forest': {
                'pros': ['Handles non-linear patterns', 'Feature importance available', 'Robust to outliers', 'No feature scaling needed'],
                'cons': ['Can overfit on small datasets', 'Slower predictions', 'Less interpretable']
            },
            'xgboost': {
                'pros': ['High accuracy', 'Handles missing values', 'Built-in regularization', 'Feature importance'],
                'cons': ['Requires hyperparameter tuning', 'Can overfit', 'Slower training on large datasets']
            },
            'lightgbm': {
                'pros': ['Very fast training', 'Memory efficient', 'High accuracy', 'Handles large datasets well'],
                'cons': ['Can overfit on small datasets', 'Sensitive to hyperparameters']
            },
            'gradient_boosting': {
                'pros': ['High accuracy', 'Handles mixed data types', 'Feature importance', 'Robust to outliers'],
                'cons': ['Slow training', 'Requires careful tuning', 'Can overfit']
            },
            'svc': {
                'pros': ['Effective in high dimensions', 'Memory efficient', 'Versatile kernels'],
                'cons': ['Slow on large datasets', 'Requires feature scaling', 'Not probabilistic by default']
            },
            'knn': {
                'pros': ['Simple and intuitive', 'No training phase', 'Works well with local patterns'],
                'cons': ['Slow predictions on large datasets', 'Sensitive to feature scaling', 'Curse of dimensionality']
            },
            'decision_tree': {
                'pros': ['Highly interpretable', 'Handles non-linear patterns', 'No feature scaling needed'],
                'cons': ['Prone to overfitting', 'Unstable with small changes', 'Biased with imbalanced data']
            },
            'mlp': {
                'pros': ['Learns complex patterns', 'Universal approximator', 'Works with non-linear data'],
                'cons': ['Requires large datasets', 'Slow training', 'Difficult to interpret', 'Many hyperparameters']
            },
            'naive_bayes': {
                'pros': ['Very fast', 'Works well with small datasets', 'Handles high dimensions'],
                'cons': ['Assumes feature independence', 'May underfit complex patterns']
            },
            'gaussian_nb': {
                'pros': ['Very fast', 'Works well with small datasets', 'Simple to implement'],
                'cons': ['Assumes Gaussian distribution', 'Assumes feature independence']
            },
            'lda': {
                'pros': ['Dimensionality reduction', 'Fast prediction', 'Interpretable'],
                'cons': ['Assumes Gaussian distribution', 'Limited to linear boundaries']
            },
            'adaboost': {
                'pros': ['Good accuracy', 'Less prone to overfitting', 'Feature importance'],
                'cons': ['Sensitive to noisy data', 'Slower training']
            },
            'extra_trees': {
                'pros': ['Faster than Random Forest', 'Reduces overfitting', 'Feature importance'],
                'cons': ['May require more trees', 'Less interpretable']
            },
            
            # Regression
            'linear_regression': {
                'pros': ['Fast and simple', 'Highly interpretable', 'Works well with linear relationships'],
                'cons': ['Assumes linearity', 'Sensitive to outliers', 'Cannot capture non-linear patterns']
            },
            'ridge': {
                'pros': ['Handles multicollinearity', 'Prevents overfitting', 'Stable predictions'],
                'cons': ['Assumes linearity', 'All features retained']
            },
            'lasso': {
                'pros': ['Feature selection', 'Prevents overfitting', 'Sparse solutions'],
                'cons': ['Assumes linearity', 'May eliminate important features']
            },
            'svr': {
                'pros': ['Effective in high dimensions', 'Robust to outliers', 'Non-linear capabilities'],
                'cons': ['Slow on large datasets', 'Requires feature scaling', 'Memory intensive']
            },
            'elasticnet': {
                'pros': ['Combines L1 and L2 regularization', 'Feature selection', 'Handles multicollinearity'],
                'cons': ['Assumes linearity', 'Requires hyperparameter tuning']
            },
            'bayesian_ridge': {
                'pros': ['Automatic relevance determination', 'Uncertainty estimates', 'Robust to overfitting'],
                'cons': ['Assumes Gaussian noise', 'Slower than Ridge']
            },
            'huber': {
                'pros': ['Robust to outliers', 'Good for noisy data', 'Combines L1 and L2 loss'],
                'cons': ['Requires tuning epsilon parameter', 'Slower convergence']
            },
            
            # Clustering
            'kmeans': {
                'pros': ['Fast and scalable', 'Simple to understand', 'Works well with spherical clusters'],
                'cons': ['Must specify k', 'Sensitive to initialization', 'Assumes spherical clusters']
            },
            'dbscan': {
                'pros': ['Finds arbitrary shapes', 'Handles noise', 'No need to specify k'],
                'cons': ['Sensitive to parameters', 'Struggles with varying densities']
            },
            'gaussian_mixture': {
                'pros': ['Soft clustering', 'Handles elliptical clusters', 'Probabilistic'],
                'cons': ['Assumes Gaussian distributions', 'Sensitive to initialization']
            },
            'agglomerative': {
                'pros': ['Hierarchical structure', 'No need to specify k upfront', 'Deterministic'],
                'cons': ['Slow on large datasets', 'Cannot undo merges']
            },
            'spectral': {
                'pros': ['Handles non-convex clusters', 'Works well with graphs', 'Effective for complex structures'],
                'cons': ['Slow on large datasets', 'Memory intensive', 'Requires k']
            },
            
            # Time Series
            'arima': {
                'pros': ['Good for stationary data', 'Captures trends and seasonality', 'Interpretable'],
                'cons': ['Requires stationary data', 'Manual parameter selection', 'Limited to univariate']
            },
            'prophet': {
                'pros': ['Handles missing data', 'Captures multiple seasonality', 'Automatic parameter selection'],
                'cons': ['May overfit on small datasets', 'Less control over model']
            },
            'sarima': {
                'pros': ['Captures seasonal patterns', 'Interpretable', 'Good for periodic data'],
                'cons': ['Requires stationary data', 'Many parameters to tune', 'Computationally expensive']
            },
            'exponential_smoothing': {
                'pros': ['Simple and fast', 'Handles trends and seasonality', 'Good for short-term forecasts'],
                'cons': ['Limited flexibility', 'Assumes additive/multiplicative patterns']
            },
            'lstm': {
                'pros': ['Captures long-term dependencies', 'Handles complex patterns', 'No stationarity required'],
                'cons': ['Requires large datasets', 'Slow training', 'Many hyperparameters', 'Black box']
            },
            'xgboost_ts': {
                'pros': ['High accuracy', 'Handles non-linear patterns', 'Fast predictions'],
                'cons': ['Requires feature engineering', 'May not capture temporal dependencies well']
            },
            'random_forest_ts': {
                'pros': ['Robust to outliers', 'Handles non-linear patterns', 'Feature importance'],
                'cons': ['Requires feature engineering', 'May not capture complex temporal patterns']
            }
        }
        
        # Default characteristics for models not in the database
        default_characteristics = {
            'pros': ['Suitable for this dataset', 'Good performance metrics', 'Established algorithm'],
            'cons': ['May require hyperparameter tuning', 'Performance depends on data characteristics']
        }
        
        # Collect all model results
        model_data = []
        for model_name, results in models_results.items():
            if results.get("status") == "completed":
                metrics = results.get("metrics", {})
                test_metrics = metrics.get("test", {})
                primary_metric = metrics.get("primary_metric", "score")
                
                # Get primary metric value
                primary_value = test_metrics.get(primary_metric, 0)
                
                # Prepare metrics display
                metrics_display = {}
                if self.task_type == 'classification':
                    metrics_display = {
                        'Accuracy': f"{test_metrics.get('accuracy', 0):.4f}",
                        'F1 Score': f"{test_metrics.get('f1_weighted', test_metrics.get('f1', 0)):.4f}",
                        'Precision': f"{test_metrics.get('precision_weighted', test_metrics.get('precision', 0)):.4f}",
                        'Recall': f"{test_metrics.get('recall_weighted', test_metrics.get('recall', 0)):.4f}",
                        'ROC-AUC': f"{test_metrics.get('roc_auc', 'N/A')}" if test_metrics.get('roc_auc') else 'N/A'
                    }
                elif self.task_type == 'regression':
                    metrics_display = {
                        'R² Score': f"{test_metrics.get('r2', 0):.4f}",
                        'RMSE': f"{test_metrics.get('rmse', 0):.4f}",
                        'MAE': f"{test_metrics.get('mae', 0):.4f}",
                        'MAPE': f"{test_metrics.get('mape', 0):.4f}" if test_metrics.get('mape') else 'N/A'
                    }
                elif self.task_type == 'clustering':
                    metrics_display = {
                        'Silhouette': f"{test_metrics.get('silhouette_score', 0):.4f}",
                        'Calinski-Harabasz': f"{test_metrics.get('calinski_harabasz_score', 0):.2f}",
                        'Davies-Bouldin': f"{test_metrics.get('davies_bouldin_score', 0):.4f}",
                        'N Clusters': test_metrics.get('n_clusters', 'N/A')
                    }
                elif self.task_type == 'time_series':
                    metrics_display = {
                        'RMSE': f"{test_metrics.get('rmse', 0):.4f}",
                        'MAE': f"{test_metrics.get('mae', 0):.4f}",
                        'MAPE': f"{test_metrics.get('mape', 0):.4f}" if test_metrics.get('mape') else 'N/A',
                        'R²': f"{test_metrics.get('r2', 0):.4f}"
                    }
                
                # Get base characteristics
                base_characteristics = model_characteristics.get(model_name, default_characteristics)
                
                # Generate intelligent pros/cons based on performance
                intelligent_pros, intelligent_cons = await self._generate_intelligent_insights(
                    model_name, test_metrics, primary_metric, primary_value, base_characteristics
                )
                
                model_data.append({
                    'model_name': model_name,
                    'primary_value': primary_value,
                    'primary_metric': primary_metric,
                    'metrics': metrics_display,
                    'advantages': intelligent_pros,
                    'disadvantages': intelligent_cons
                })
        
        # Sort by primary metric (higher is better for most metrics except error metrics)
        higher_is_better = self.task_type in ['classification', 'clustering'] or \
                          (self.task_type == 'regression' and 'r2' in str(model_data[0]['primary_metric']).lower() if model_data else False)
        
        model_data.sort(key=lambda x: x['primary_value'], reverse=higher_is_better)
        
        # Select top 6-8 models
        top_models = model_data[:min(8, len(model_data))]
        
        # Create comparison table
        comparison_table = []
        for rank, model in enumerate(top_models, 1):
            # Calculate recommendation score (0-100)
            if higher_is_better:
                normalized_score = (model['primary_value'] - min(m['primary_value'] for m in top_models)) / \
                                  (max(m['primary_value'] for m in top_models) - min(m['primary_value'] for m in top_models) + 1e-10)
            else:
                normalized_score = (max(m['primary_value'] for m in top_models) - model['primary_value']) / \
                                  (max(m['primary_value'] for m in top_models) - min(m['primary_value'] for m in top_models) + 1e-10)
            
            recommendation_score = 60 + (normalized_score * 40)  # Scale to 60-100
            
            # Add rank badge
            rank_badge = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
            
            comparison_table.append({
                'rank': rank,
                'rank_badge': rank_badge,
                'model_name': model['model_name'].replace('_', ' ').title(),
                'metrics': model['metrics'],
                'advantages': model['advantages'][:4],  # Top 4 advantages
                'disadvantages': model['disadvantages'][:3],  # Top 3 disadvantages
                'recommendation_score': round(recommendation_score, 1),
                'primary_metric_name': model['primary_metric'],
                'primary_metric_value': round(model['primary_value'], 4)
            })
        
        # Generate recommendations for top 3-4 models
        recommendations = await self._generate_model_recommendations(comparison_table[:min(4, len(comparison_table))])
        
        return {
            'comparison_table': comparison_table,
            'total_models_evaluated': len(model_data),
            'top_model': comparison_table[0] if comparison_table else None,
            'task_type': self.task_type,
            'summary': f"Evaluated {len(model_data)} models. Top performer: {comparison_table[0]['model_name'] if comparison_table else 'None'}",
            'recommendations': recommendations
        }
    
    async def _generate_model_recommendations(self, top_models: List[Dict]) -> Dict[str, Any]:
        """
        Generate plain language recommendations for top 3-4 models with detailed explanations
        """
        if not top_models:
            return {
                'recommended_models': [],
                'overall_recommendation': 'No models available for recommendation.',
                'decision_guide': {}
            }
        
        recommendations = {
            'recommended_models': [],
            'overall_recommendation': '',
            'decision_guide': {},
            'executive_summary': ''
        }
        
        # Get dataset info
        n_samples = len(self.df) if hasattr(self, 'df') and self.df is not None else 0
        n_features = self.df.shape[1] if hasattr(self, 'df') and self.df is not None and len(self.df.shape) > 1 else 0
        
        # Generate recommendations for each top model
        for i, model in enumerate(top_models, 1):
            model_name = model['model_name']
            model_name_lower = model_name.lower()
            rank = model['rank']
            score = model['recommendation_score']
            metrics = model['metrics']
            
            # Generate why this model performed best
            performance_explanation = self._explain_performance(model, self.task_type, rank)
            
            # Generate trade-offs
            trade_offs = self._explain_tradeoffs(model_name_lower, model, self.task_type)
            
            # Generate use case suitability
            best_for = self._explain_use_cases(model_name_lower, n_samples, n_features, self.task_type)
            
            # Generate deployment considerations
            deployment_notes = self._explain_deployment(model_name_lower, n_samples)
            
            recommendation = {
                'rank': rank,
                'model_name': model_name,
                'recommendation_score': score,
                'why_it_performed_best': performance_explanation,
                'key_trade_offs': trade_offs,
                'best_suited_for': best_for,
                'deployment_considerations': deployment_notes,
                'metrics_summary': self._format_metrics_summary(metrics, self.task_type)
            }
            
            recommendations['recommended_models'].append(recommendation)
        
        # Generate overall recommendation
        best_model = top_models[0]
        recommendations['overall_recommendation'] = self._generate_overall_recommendation(
            best_model, top_models, self.task_type, n_samples, n_features
        )
        
        # Generate executive summary
        recommendations['executive_summary'] = self._generate_executive_summary(
            top_models, self.task_type, n_samples
        )
        
        # Generate decision guide
        recommendations['decision_guide'] = self._generate_decision_guide(top_models, self.task_type)
        
        return recommendations
    
    def _explain_performance(self, model: Dict, task_type: str, rank: int) -> str:
        """Explain in plain language why this model performed well"""
        model_name = model['model_name'].lower()
        metrics = model['metrics']
        score = model['recommendation_score']
        
        explanations = []
        
        if rank == 1:
            explanations.append(f"This model achieved the highest performance score ({score:.1f}/100) among all tested algorithms.")
        else:
            explanations.append(f"This model secured rank #{rank} with a strong performance score of {score:.1f}/100.")
        
        # Task-specific performance explanation
        if task_type == 'classification':
            acc = metrics.get('Accuracy', '0')
            f1 = metrics.get('F1 Score', '0')
            explanations.append(f"It demonstrated excellent predictive accuracy ({acc}) with well-balanced precision and recall ({f1} F1 score).")
            
            if 'forest' in model_name or 'xgb' in model_name:
                explanations.append("The ensemble approach effectively captured complex non-linear patterns in the data, leading to robust predictions across different classes.")
            elif 'logistic' in model_name:
                explanations.append("The linear decision boundaries aligned well with the data structure, providing fast and interpretable predictions.")
            elif 'neural' in model_name or 'mlp' in model_name:
                explanations.append("The neural architecture successfully learned intricate patterns and interactions between features.")
                
        elif task_type == 'regression':
            r2 = metrics.get('R² Score', '0')
            rmse = metrics.get('RMSE', '0')
            explanations.append(f"It explained {float(r2)*100:.1f}% of the variance in the target variable with low prediction error (RMSE: {rmse}).")
            
            if 'forest' in model_name or 'xgb' in model_name or 'gradient' in model_name:
                explanations.append("The boosting/ensemble technique effectively modeled non-linear relationships and interactions between predictors.")
            elif 'linear' in model_name or 'ridge' in model_name or 'lasso' in model_name:
                explanations.append("The regularized linear approach provided stable predictions while preventing overfitting.")
                
        elif task_type == 'clustering':
            sil = metrics.get('Silhouette', '0')
            explanations.append(f"It created well-separated and cohesive clusters (Silhouette: {sil}), indicating meaningful groupings in the data.")
            
        elif task_type == 'time_series':
            rmse = metrics.get('RMSE', '0')
            mape = metrics.get('MAPE', 'N/A')
            explanations.append(f"It accurately captured temporal patterns with low forecast error (RMSE: {rmse}, MAPE: {mape}).")
        
        return ' '.join(explanations)
    
    def _explain_tradeoffs(self, model_name: str, model: Dict, task_type: str) -> Dict[str, str]:
        """Explain the trade-offs for this model"""
        trade_offs = {
            'speed_vs_accuracy': '',
            'interpretability_vs_performance': '',
            'complexity_vs_maintainability': ''
        }
        
        # Speed vs Accuracy
        if 'xgb' in model_name or 'lightgbm' in model_name:
            trade_offs['speed_vs_accuracy'] = "Excellent balance: Fast training and prediction while maintaining high accuracy. Best of both worlds for most use cases."
        elif 'forest' in model_name:
            trade_offs['speed_vs_accuracy'] = "Slightly slower predictions due to multiple decision trees, but the accuracy gains justify the computational cost."
        elif 'svc' in model_name or 'svr' in model_name:
            trade_offs['speed_vs_accuracy'] = "Slower on large datasets but provides strong performance. Consider faster alternatives for real-time applications."
        elif 'linear' in model_name or 'logistic' in model_name:
            trade_offs['speed_vs_accuracy'] = "Extremely fast training and prediction. May sacrifice some accuracy for speed, ideal for large-scale deployments."
        elif 'neural' in model_name or 'mlp' in model_name or 'lstm' in model_name:
            trade_offs['speed_vs_accuracy'] = "Slower training time but can capture very complex patterns. Best when high accuracy is critical and computational resources are available."
        else:
            trade_offs['speed_vs_accuracy'] = "Balanced approach suitable for most production environments."
        
        # Interpretability vs Performance
        if 'linear' in model_name or 'logistic' in model_name or 'lasso' in model_name or 'ridge' in model_name:
            trade_offs['interpretability_vs_performance'] = "Highly interpretable: You can easily understand how each feature influences predictions. Perfect for regulated industries or when explainability is required."
        elif 'tree' in model_name and 'forest' not in model_name:
            trade_offs['interpretability_vs_performance'] = "Very interpretable: Decision paths are easy to visualize and explain. Good for business stakeholders."
        elif 'forest' in model_name or 'xgb' in model_name or 'gradient' in model_name:
            trade_offs['interpretability_vs_performance'] = "Moderate interpretability: Feature importance is available, but individual predictions are harder to explain. Prioritizes performance over explainability."
        elif 'neural' in model_name or 'mlp' in model_name or 'lstm' in model_name:
            trade_offs['interpretability_vs_performance'] = "Black box model: Highest potential performance but difficult to interpret. Use SHAP or LIME for post-hoc explanations if needed."
        else:
            trade_offs['interpretability_vs_performance'] = "Moderate interpretability with good performance characteristics."
        
        # Complexity vs Maintainability
        if 'linear' in model_name or 'logistic' in model_name or 'naive' in model_name:
            trade_offs['complexity_vs_maintainability'] = "Simple and easy to maintain: Minimal hyperparameters, easy to retrain, and straightforward to deploy."
        elif 'forest' in model_name:
            trade_offs['complexity_vs_maintainability'] = "Low maintenance: Few hyperparameters to tune, robust to different data types, and easy to update."
        elif 'xgb' in model_name or 'lightgbm' in model_name:
            trade_offs['complexity_vs_maintainability'] = "Moderate complexity: Requires some hyperparameter tuning but offers excellent performance. Well-documented libraries make deployment easier."
        elif 'neural' in model_name or 'mlp' in model_name or 'lstm' in model_name:
            trade_offs['complexity_vs_maintainability'] = "High complexity: Many hyperparameters, requires careful tuning, and may need retraining. Best when you have ML engineering resources."
        else:
            trade_offs['complexity_vs_maintainability'] = "Moderate complexity with standard maintenance requirements."
        
        return trade_offs
    
    def _explain_use_cases(self, model_name: str, n_samples: int, n_features: int, task_type: str) -> List[str]:
        """Explain when this model is most suitable"""
        use_cases = []
        
        # Dataset size considerations
        if n_samples < 1000:
            if 'linear' in model_name or 'logistic' in model_name or 'naive' in model_name:
                use_cases.append("✓ Ideal for small datasets where simpler models generalize better")
            elif 'forest' in model_name and n_samples > 500:
                use_cases.append("✓ Good for small to medium datasets with sufficient data for ensemble methods")
        else:
            if 'xgb' in model_name or 'lightgbm' in model_name:
                use_cases.append("✓ Excellent for large datasets where performance is critical")
            elif 'forest' in model_name:
                use_cases.append("✓ Scales well with your dataset size while maintaining accuracy")
        
        # Feature space considerations
        if n_features > 50:
            if 'forest' in model_name or 'xgb' in model_name or 'lightgbm' in model_name:
                use_cases.append("✓ Handles high-dimensional data effectively without feature selection")
            elif 'lasso' in model_name:
                use_cases.append("✓ Performs automatic feature selection in high-dimensional spaces")
        
        # Business context
        if 'linear' in model_name or 'logistic' in model_name:
            use_cases.append("✓ Best when model interpretability is legally or ethically required")
            use_cases.append("✓ Ideal for regulated industries (finance, healthcare, insurance)")
        
        if 'forest' in model_name or 'xgb' in model_name:
            use_cases.append("✓ Perfect when predictive accuracy is the top priority")
            use_cases.append("✓ Suitable for competitive scenarios (Kaggle, business competitions)")
        
        if 'neural' in model_name or 'mlp' in model_name:
            use_cases.append("✓ Best for highly complex, non-linear patterns")
            use_cases.append("✓ Suitable when you have ample training data and computational resources")
        
        # Real-time considerations
        if 'linear' in model_name or 'logistic' in model_name or 'naive' in model_name:
            use_cases.append("✓ Excellent for real-time predictions with low latency requirements")
        elif 'lightgbm' in model_name:
            use_cases.append("✓ Good balance for real-time systems needing both speed and accuracy")
        
        # Task-specific use cases
        if task_type == 'classification':
            if 'xgb' in model_name or 'lightgbm' in model_name:
                use_cases.append("✓ Strong for imbalanced classification problems")
        elif task_type == 'time_series':
            if 'arima' in model_name or 'sarima' in model_name:
                use_cases.append("✓ Best for short to medium-term forecasting with clear trends")
            elif 'prophet' in model_name:
                use_cases.append("✓ Ideal for business forecasting with multiple seasonality patterns")
            elif 'lstm' in model_name:
                use_cases.append("✓ Perfect for long-term dependencies and complex temporal patterns")
        
        return use_cases[:5]  # Return top 5 use cases
    
    def _explain_deployment(self, model_name: str, n_samples: int) -> Dict[str, str]:
        """Explain deployment considerations"""
        deployment = {
            'production_readiness': '',
            'resource_requirements': '',
            'monitoring_needs': ''
        }
        
        # Production readiness
        if 'xgb' in model_name or 'lightgbm' in model_name or 'forest' in model_name:
            deployment['production_readiness'] = "Production-ready: Well-tested libraries with strong community support. Easy to serialize and deploy."
        elif 'linear' in model_name or 'logistic' in model_name:
            deployment['production_readiness'] = "Highly production-ready: Minimal dependencies, fast deployment, and stable performance."
        elif 'neural' in model_name or 'mlp' in model_name:
            deployment['production_readiness'] = "Requires careful deployment: Need to manage model versioning, input validation, and potential GPU dependencies."
        else:
            deployment['production_readiness'] = "Standard deployment process with common ML frameworks."
        
        # Resource requirements
        if n_samples > 100000:
            if 'lightgbm' in model_name:
                deployment['resource_requirements'] = "Low memory footprint: Efficient for large-scale deployments. CPU-only deployment is sufficient."
            elif 'forest' in model_name or 'xgb' in model_name:
                deployment['resource_requirements'] = "Moderate resources: May need 2-4GB RAM for model serving. Consider model compression for edge deployment."
            elif 'svc' in model_name or 'svr' in model_name:
                deployment['resource_requirements'] = "Higher memory requirements: Consider model size when deploying at scale."
        else:
            deployment['resource_requirements'] = "Minimal resource requirements: Can run on standard cloud instances or even edge devices."
        
        # Monitoring needs
        if 'linear' in model_name or 'logistic' in model_name:
            deployment['monitoring_needs'] = "Simple monitoring: Track prediction accuracy and feature distributions. Easy to detect drift."
        elif 'forest' in model_name or 'xgb' in model_name:
            deployment['monitoring_needs'] = "Standard monitoring: Monitor feature importance changes, prediction distributions, and model performance metrics."
        elif 'neural' in model_name:
            deployment['monitoring_needs'] = "Comprehensive monitoring: Track activation patterns, gradients, and layer outputs for debugging."
        else:
            deployment['monitoring_needs'] = "Standard ML monitoring practices apply."
        
        return deployment
    
    def _format_metrics_summary(self, metrics: Dict, task_type: str) -> str:
        """Create a plain language metrics summary"""
        if task_type == 'classification':
            return f"Achieved {metrics.get('Accuracy', 'N/A')} accuracy with {metrics.get('F1 Score', 'N/A')} F1 score, demonstrating strong predictive capability."
        elif task_type == 'regression':
            return f"Explained variance: {metrics.get('R² Score', 'N/A')} | Prediction error (RMSE): {metrics.get('RMSE', 'N/A')}"
        elif task_type == 'clustering':
            return f"Cluster quality (Silhouette): {metrics.get('Silhouette', 'N/A')} | Found {metrics.get('N Clusters', 'N/A')} distinct groups"
        elif task_type == 'time_series':
            return f"Forecast accuracy (MAPE): {metrics.get('MAPE', 'N/A')} | Prediction error (RMSE): {metrics.get('RMSE', 'N/A')}"
        return "Metrics available in detailed view"
    
    def _generate_overall_recommendation(self, best_model: Dict, top_models: List[Dict], 
                                        task_type: str, n_samples: int, n_features: int) -> str:
        """Generate overall recommendation paragraph"""
        model_name = best_model['model_name']
        score = best_model['recommendation_score']
        
        recommendation = f"**Primary Recommendation: {model_name}**\n\n"
        recommendation += f"Based on comprehensive evaluation of {len(top_models)} top-performing models, "
        recommendation += f"{model_name} is the recommended choice with a score of {score:.1f}/100. "
        
        # Add context-specific reasoning
        if score > 95:
            recommendation += "This model significantly outperformed alternatives across all key metrics. "
        elif score > 85:
            recommendation += "This model demonstrated strong performance with a good balance of accuracy and efficiency. "
        else:
            recommendation += "While multiple models performed similarly, this model offers the best trade-offs for your use case. "
        
        # Add alternative recommendation
        if len(top_models) > 1:
            second_best = top_models[1]
            recommendation += f"\n\n**Alternative Option: {second_best['model_name']}** (Score: {second_best['recommendation_score']:.1f}/100) "
            recommendation += f"Consider this as a fallback if {model_name} doesn't meet deployment constraints or if you need different trade-offs."
        
        return recommendation
    
    def _generate_executive_summary(self, top_models: List[Dict], task_type: str, n_samples: int) -> str:
        """Generate executive summary"""
        summary = f"## Executive Summary\n\n"
        summary += f"Evaluated {len(top_models)} high-performing {task_type} models on your dataset ({n_samples:,} samples).\n\n"
        
        best = top_models[0]
        summary += f"**Winner:** {best['model_name']} achieved the best performance "
        summary += f"({best['recommendation_score']:.1f}/100 recommendation score).\n\n"
        
        summary += f"**Key Insight:** "
        if 'xgb' in best['model_name'].lower() or 'lightgbm' in best['model_name'].lower():
            summary += "Gradient boosting methods proved most effective for your data's complexity. "
        elif 'forest' in best['model_name'].lower():
            summary += "Ensemble methods effectively captured the non-linear patterns in your data. "
        elif 'linear' in best['model_name'].lower() or 'logistic' in best['model_name'].lower():
            summary += "Linear models provided the best balance of performance and interpretability. "
        elif 'neural' in best['model_name'].lower():
            summary += "Deep learning successfully modeled the complex patterns in your dataset. "
        
        summary += f"The top {len(top_models)} models are production-ready and suitable for immediate deployment."
        
        return summary
    
    def _generate_decision_guide(self, top_models: List[Dict], task_type: str) -> Dict[str, str]:
        """Generate a decision guide for choosing between top models"""
        guide = {
            'prioritize_accuracy': '',
            'prioritize_speed': '',
            'prioritize_interpretability': '',
            'prioritize_simplicity': ''
        }
        
        model_names = [m['model_name'].lower() for m in top_models]
        
        # Find best for each priority
        for name, model in zip(model_names, top_models):
            if 'xgb' in name or 'lightgbm' in name or 'gradient' in name:
                if not guide['prioritize_accuracy']:
                    guide['prioritize_accuracy'] = f"Choose **{model['model_name']}** - Highest accuracy among ensemble methods"
            
            if 'lightgbm' in name or 'linear' in name or 'logistic' in name:
                if not guide['prioritize_speed']:
                    guide['prioritize_speed'] = f"Choose **{model['model_name']}** - Fastest training and prediction"
            
            if 'linear' in name or 'logistic' in name or ('tree' in name and 'forest' not in name):
                if not guide['prioritize_interpretability']:
                    guide['prioritize_interpretability'] = f"Choose **{model['model_name']}** - Most interpretable predictions"
            
            if 'linear' in name or 'logistic' in name or 'naive' in name:
                if not guide['prioritize_simplicity']:
                    guide['prioritize_simplicity'] = f"Choose **{model['model_name']}** - Easiest to deploy and maintain"
        
        # Fill in blanks with top model
        for key in guide:
            if not guide[key]:
                guide[key] = f"Choose **{top_models[0]['model_name']}** - Best overall performer"
        
        return guide
    
    async def predict(self, model_path: str, input_data: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions using a trained model"""
        
        try:
            # Load model
            model = joblib.load(model_path)
            
            # Make predictions
            predictions = model.predict(input_data)
            
            # Get prediction probabilities for classification
            probabilities = None
            if self.task_type == 'classification' and hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_data)
            
            return {
                "predictions": predictions.tolist(),
                "probabilities": probabilities.tolist() if probabilities is not None else None,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
