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
        
    async def auto_train_models(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """
        Automatically detect task type and train multiple models
        
        Args:
            df: Input dataframe
            target_column: Target column name (auto-detected if None)
            
        Returns:
            Dictionary with training results and model comparisons
        """
        
        logger.info("Starting automated model training")
        
        # Auto-detect target column if not provided
        if target_column is None:
            target_column = await self._auto_detect_target(df)
            if target_column is None:
                raise ValueError("Could not auto-detect target column. Please specify target_column.")
        
        self.target_column = target_column
        
        # Detect task type
        self.task_type = await self._detect_task_type(df, target_column)
        logger.info(f"Detected task type: {self.task_type}")
        
        # Prepare features and target
        X, y = await self._prepare_features_target(df, target_column)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=settings.RANDOM_STATE,
            stratify=y if self.task_type == 'classification' and len(np.unique(y)) > 1 else None
        )
        
        # Create preprocessing pipeline
        self.preprocessor = await self._create_preprocessor(X_train)
        
        # Train multiple models
        models_to_train = await self._get_models_for_task(self.task_type)
        
        results = {
            "task_type": self.task_type,
            "target_column": target_column,
            "feature_columns": self.feature_columns,
            "dataset_info": {
                "total_samples": len(df),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "n_features": X_train.shape[1]
            },
            "models": {},
            "best_model": None,
            "model_comparison": {}
        }
        
        best_score = -np.inf if self.task_type == 'classification' else np.inf
        best_model_name = None
        
        for model_name, model_config in models_to_train.items():
            logger.info(f"Training {model_name}")
            
            try:
                model_results = await self._train_single_model(
                    model_name, model_config, X_train, X_test, y_train, y_test
                )
                
                results["models"][model_name] = model_results
                
                # Track best model
                primary_metric = model_results["metrics"]["primary_metric"]
                score = model_results["metrics"][primary_metric]
                
                is_better = (
                    (self.task_type == 'classification' and score > best_score) or
                    (self.task_type == 'regression' and score < best_score)
                )
                
                if is_better:
                    best_score = score
                    best_model_name = model_name
                    
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                results["models"][model_name] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        # Set best model
        if best_model_name:
            results["best_model"] = best_model_name
            results["model_comparison"] = await self._create_model_comparison(results["models"])
        
        logger.info("Model training completed")
        return results
    
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
    
    async def _detect_task_type(self, df: pd.DataFrame, target_column: str) -> str:
        """Detect if task is classification or regression"""
        
        target_series = df[target_column]
        
        # Check data type
        if target_series.dtype in ['object', 'category']:
            return 'classification'
        
        # Check unique values ratio for numeric columns
        unique_ratio = target_series.nunique() / len(target_series)
        
        if unique_ratio < 0.05 or target_series.nunique() <= 20:
            return 'classification'
        else:
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
    
    async def _get_models_for_task(self, task_type: str) -> Dict[str, Dict[str, Any]]:
        """Get appropriate models for the task type"""
        
        if task_type == 'classification':
            models = {
                'random_forest': {
                    'model': RandomForestClassifier(random_state=settings.RANDOM_STATE),
                    'params': {
                        'model__n_estimators': [100, 200],
                        'model__max_depth': [10, 20, None],
                        'model__min_samples_split': [2, 5]
                    }
                },
                'logistic_regression': {
                    'model': LogisticRegression(random_state=settings.RANDOM_STATE, max_iter=1000),
                    'params': {
                        'model__C': [0.1, 1.0, 10.0],
                        'model__penalty': ['l1', 'l2']
                    }
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
                    }
                }
        
        else:  # regression
            models = {
                'random_forest': {
                    'model': RandomForestRegressor(random_state=settings.RANDOM_STATE),
                    'params': {
                        'model__n_estimators': [100, 200],
                        'model__max_depth': [10, 20, None],
                        'model__min_samples_split': [2, 5]
                    }
                },
                'linear_regression': {
                    'model': LinearRegression(),
                    'params': {}
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
                    }
                }
        
        return models
    
    async def _train_single_model(self, model_name: str, model_config: Dict[str, Any],
                                X_train: pd.DataFrame, X_test: pd.DataFrame,
                                y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """Train a single model with hyperparameter optimization"""
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('model', model_config['model'])
        ])
        
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
        metrics = await self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test, pipeline, X_test)
        
        # Save model
        model_id = str(uuid.uuid4())
        model_path = settings.MODELS_DIR / f"{model_name}_{model_id}.joblib"
        joblib.dump(pipeline, model_path)
        
        return {
            "model_name": model_name,
            "model_id": model_id,
            "model_path": str(model_path),
            "metrics": metrics,
            "status": "completed",
            "trained_at": datetime.utcnow().isoformat()
        }
    
    async def _optimize_hyperparameters(self, pipeline: Pipeline, param_grid: Dict[str, List],
                                      X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        
        def objective(trial):
            # Sample parameters
            params = {}
            for param, values in param_grid.items():
                if isinstance(values[0], int):
                    params[param] = trial.suggest_int(param, min(values), max(values))
                elif isinstance(values[0], float):
                    params[param] = trial.suggest_float(param, min(values), max(values))
                else:
                    params[param] = trial.suggest_categorical(param, values)
            
            # Set parameters
            temp_pipeline = Pipeline([
                ('preprocessor', pipeline.named_steps['preprocessor']),
                ('model', pipeline.named_steps['model'].__class__(**{
                    k.replace('model__', ''): v for k, v in params.items()
                }))
            ])
            
            # Cross-validation
            if self.task_type == 'classification':
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.RANDOM_STATE)
                scores = cross_val_score(temp_pipeline, X_train, y_train, cv=cv, scoring='accuracy')
            else:
                cv = KFold(n_splits=3, shuffle=True, random_state=settings.RANDOM_STATE)
                scores = cross_val_score(temp_pipeline, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
            
            return scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20, timeout=300)  # 5 minutes max
        
        return study.best_params
    
    async def _calculate_metrics(self, y_train: pd.Series, y_pred_train: np.ndarray,
                               y_test: pd.Series, y_pred_test: np.ndarray,
                               pipeline: Pipeline, X_test: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive metrics"""
        
        metrics = {}
        
        if self.task_type == 'classification':
            # Training metrics
            metrics['train'] = {
                'accuracy': float(accuracy_score(y_train, y_pred_train)),
                'precision': float(precision_score(y_train, y_pred_train, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_train, y_pred_train, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_train, y_pred_train, average='weighted', zero_division=0))
            }
            
            # Test metrics
            metrics['test'] = {
                'accuracy': float(accuracy_score(y_test, y_pred_test)),
                'precision': float(precision_score(y_test, y_pred_test, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_test, y_pred_test, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_test, y_pred_test, average='weighted', zero_division=0))
            }
            
            # ROC AUC for binary classification
            if len(np.unique(y_test)) == 2:
                try:
                    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                    metrics['test']['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba))
                except:
                    pass
            
            metrics['primary_metric'] = 'accuracy'
            
        else:  # regression
            # Training metrics
            metrics['train'] = {
                'mse': float(mean_squared_error(y_train, y_pred_train)),
                'mae': float(mean_absolute_error(y_train, y_pred_train)),
                'r2': float(r2_score(y_train, y_pred_train)),
                'rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
            }
            
            # Test metrics
            metrics['test'] = {
                'mse': float(mean_squared_error(y_test, y_pred_test)),
                'mae': float(mean_absolute_error(y_test, y_pred_test)),
                'r2': float(r2_score(y_test, y_pred_test)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
            }
            
            metrics['primary_metric'] = 'rmse'
        
        return metrics
    
    async def _create_model_comparison(self, models_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create model comparison summary"""
        
        comparison = {
            "summary_table": [],
            "best_performers": {},
            "recommendations": []
        }
        
        # Create summary table
        for model_name, results in models_results.items():
            if results.get("status") == "completed":
                metrics = results["metrics"]
                
                row = {
                    "model": model_name,
                    "primary_metric": metrics["test"][metrics["primary_metric"]]
                }
                
                # Add key metrics
                if self.task_type == 'classification':
                    row.update({
                        "accuracy": metrics["test"]["accuracy"],
                        "f1_score": metrics["test"]["f1_score"],
                        "precision": metrics["test"]["precision"],
                        "recall": metrics["test"]["recall"]
                    })
                else:
                    row.update({
                        "rmse": metrics["test"]["rmse"],
                        "mae": metrics["test"]["mae"],
                        "r2": metrics["test"]["r2"]
                    })
                
                comparison["summary_table"].append(row)
        
        # Sort by primary metric
        if comparison["summary_table"]:
            reverse_sort = self.task_type == 'classification'  # Higher is better for classification
            comparison["summary_table"].sort(
                key=lambda x: x["primary_metric"], 
                reverse=reverse_sort
            )
        
        return comparison
    
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
