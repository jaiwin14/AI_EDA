import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
from datetime import datetime
import logging

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, log_loss, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
)

# Models
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Visualization
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class BaselineAgent:
    """Agent for training baseline models and computing metrics"""
    
    def __init__(self):
        self.name = "Baseline & Metrics Agent"
        self.models = {}
        self.preprocessor = None
        self.results = {}
        
    def process(self, context) -> Dict[str, Any]:
        """
        Train baseline models and compute metrics
        
        Args:
            context: DatasetContext with target and task type
            
        Returns:
            Dict with model results, metrics, and visualizations
        """
        if not context.target_column:
            return {
                'status': 'error',
                'error': 'No target column specified',
                'models': {},
                'metrics': {},
                'best_model': None
            }
            
        if not context.task_type:
            return {
                'status': 'error', 
                'error': 'No task type specified',
                'models': {},
                'metrics': {},
                'best_model': None
            }
        
        try:
            df = context.df
            target_col = context.target_column
            task_type = context.task_type
            
            # Prepare data
            X, y = self._prepare_data(df, target_col)
            
            # Split data
            X_train, X_test, y_train, y_test = self._split_data(X, y, task_type)
            
            # Build preprocessor
            self.preprocessor = self._build_preprocessor(X_train)
            
            # Get models for task type
            models = self._get_models(task_type)
            
            # Train models
            trained_models = {}
            model_metrics = {}
            
            for name, model in models.items():
                logger.info(f"Training {name}...")
                
                try:
                    # Create pipeline
                    pipeline = Pipeline([
                        ('preprocessor', self.preprocessor),
                        ('model', model)
                    ])
                    
                    # Train
                    start_time = datetime.now()
                    pipeline.fit(X_train, y_train)
                    training_time = (datetime.now() - start_time).total_seconds()
                    
                    # Predictions
                    y_pred = pipeline.predict(X_test)
                    if hasattr(pipeline, "predict_proba") and task_type == 'classification':
                        try:
                            y_proba = pipeline.predict_proba(X_test)
                        except:
                            y_proba = None
                    else:
                        y_proba = None
                    
                    # Calculate metrics
                    metrics = self._calculate_metrics(
                        y_test, y_pred, y_proba, task_type
                    )
                    metrics['training_time'] = training_time
                    
                    # Cross-validation
                    cv_score = self._cross_validate(pipeline, X_train, y_train, task_type)
                    metrics['cv_score'] = cv_score
                    
                    trained_models[name] = {
                        'pipeline': pipeline,
                        'predictions': y_pred,
                        'probabilities': y_proba,
                        'training_time': training_time
                    }
                    model_metrics[name] = metrics
                    
                    logger.info(f"✅ {name} completed - Score: {metrics.get('primary_score', 0):.3f}")
                    
                except Exception as e:
                    logger.error(f"❌ {name} failed: {str(e)}")
                    model_metrics[name] = {
                        'error': str(e),
                        'primary_score': 0.0
                    }
            
            # Select best model
            best_model_name = self._select_best_model(model_metrics, task_type)
            
            # Generate visualizations
            visualizations = self._create_visualizations(
                trained_models, model_metrics, X_test, y_test, task_type
            )
            
            results = {
                'status': 'success',
                'task_type': task_type,
                'data_split': {
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'features': len(X.columns)
                },
                'models': model_metrics,
                'best_model': best_model_name,
                'trained_pipelines': trained_models,
                'visualizations': visualizations,
                'preprocessing_info': self._get_preprocessing_info()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Baseline agent failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'models': {},
                'metrics': {},
                'best_model': None
            }
    
    def _prepare_data(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target"""
        # Features (all columns except target)
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Remove rows with missing target
        mask = ~y.isnull()
        X = X[mask]
        y = y[mask]
        
        return X, y
    
    def _split_data(self, X: pd.DataFrame, y: pd.Series, task_type: str) -> Tuple:
        """Split data into train/test with appropriate strategy"""
        test_size = 0.2
        random_state = 42
        
        if task_type == 'classification' and y.nunique() > 1:
            # Stratified split for classification
            return train_test_split(
                X, y, test_size=test_size, random_state=random_state, 
                stratify=y
            )
        else:
            # Regular split for regression
            return train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
    
    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Build preprocessing pipeline"""
        # Identify column types
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Numeric preprocessing
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical preprocessing
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=20))
        ])
        
        # Combine preprocessors
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='drop'
        )
        
        return preprocessor
    
    def _get_models(self, task_type: str) -> Dict[str, Any]:
        """Get baseline models for the task type"""
        if task_type == 'classification':
            return {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
                'Naive Bayes': GaussianNB(),
                'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
            }
        else:  # regression
            return {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0, random_state=42),
                'Lasso Regression': Lasso(alpha=0.1, random_state=42, max_iter=1000),
                'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
                'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5)
            }
    
    def _calculate_metrics(self, y_true, y_pred, y_proba, task_type: str) -> Dict[str, float]:
        """Calculate appropriate metrics for task type"""
        metrics = {}
        
        if task_type == 'classification':
            # Basic metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
            
            # Handle binary vs multiclass
            if len(np.unique(y_true)) == 2:
                # Binary classification
                metrics['f1'] = f1_score(y_true, y_pred)
                metrics['precision'] = precision_score(y_true, y_pred)
                metrics['recall'] = recall_score(y_true, y_pred)
                
                if y_proba is not None:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                    metrics['pr_auc'] = average_precision_score(y_true, y_proba[:, 1])
                    metrics['log_loss'] = log_loss(y_true, y_proba)
            else:
                # Multiclass
                metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
                metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
                metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
                metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
                
                if y_proba is not None:
                    try:
                        metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
                        metrics['log_loss'] = log_loss(y_true, y_proba)
                    except:
                        pass
            
            # Primary score for model selection
            metrics['primary_score'] = metrics['balanced_accuracy']
            
        else:  # regression
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics['r2'] = r2_score(y_true, y_pred)
            
            # MAPE with zero protection
            mask = y_true != 0
            if mask.sum() > 0:
                metrics['mape'] = mean_absolute_percentage_error(y_true[mask], y_pred[mask])
            else:
                metrics['mape'] = np.inf
                
            # Primary score for model selection (negative RMSE for maximization)
            metrics['primary_score'] = -metrics['rmse']
        
        return metrics
    
    def _cross_validate(self, pipeline, X, y, task_type: str) -> float:
        """Perform cross-validation"""
        try:
            if task_type == 'classification':
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scoring = 'balanced_accuracy'
            else:
                cv = KFold(n_splits=5, shuffle=True, random_state=42) 
                scoring = 'neg_root_mean_squared_error'
            
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
            return scores.mean()
            
        except Exception as e:
            logger.warning(f"Cross-validation failed: {str(e)}")
            return 0.0
    
    def _select_best_model(self, model_metrics: Dict, task_type: str) -> Optional[str]:
        """Select best model based on primary score"""
        valid_models = {
            name: metrics for name, metrics in model_metrics.items() 
            if 'error' not in metrics and 'primary_score' in metrics
        }
        
        if not valid_models:
            return None
            
        best_model = max(valid_models.items(), key=lambda x: x[1]['primary_score'])
        return best_model[0]
    
    def _create_visualizations(self, trained_models: Dict, model_metrics: Dict, 
                             X_test: pd.DataFrame, y_test: pd.Series, task_type: str) -> Dict:
        """Create visualizations for model results"""
        visualizations = {}
        
        # Model comparison chart
        model_comparison_fig = self._create_model_comparison_chart(model_metrics, task_type)
        visualizations['model_comparison'] = model_comparison_fig
        
        # Get best model for detailed visualizations
        valid_models = {name: model for name, model in trained_models.items() 
                       if 'error' not in model_metrics[name]}
        
        if valid_models:
            best_model_name = max(model_metrics.items(), 
                                key=lambda x: x[1].get('primary_score', 0))[0]
            
            if best_model_name in valid_models:
                best_model_info = valid_models[best_model_name]
                
                if task_type == 'classification':
                    # Confusion matrix
                    cm_fig = self._create_confusion_matrix(
                        y_test, best_model_info['predictions'], best_model_name
                    )
                    visualizations['confusion_matrix'] = cm_fig
                    
                    # ROC curve if binary and probabilities available
                    if (len(np.unique(y_test)) == 2 and 
                        best_model_info['probabilities'] is not None):
                        roc_fig = self._create_roc_curve(
                            y_test, best_model_info['probabilities'][:, 1], best_model_name
                        )
                        visualizations['roc_curve'] = roc_fig
                        
                else:  # regression
                    # Residual plots
                    residual_fig = self._create_residual_plots(
                        y_test, best_model_info['predictions'], best_model_name
                    )
                    visualizations['residual_plots'] = residual_fig
        
        return visualizations
    
    def _create_model_comparison_chart(self, model_metrics: Dict, task_type: str) -> go.Figure:
        """Create model comparison bar chart"""
        models = []
        scores = []
        
        for name, metrics in model_metrics.items():
            if 'error' not in metrics and 'primary_score' in metrics:
                models.append(name)
                scores.append(metrics['primary_score'])
        
        if task_type == 'classification':
            title = "Model Comparison - Balanced Accuracy"
            y_label = "Balanced Accuracy"
        else:
            title = "Model Comparison - RMSE (lower is better)"
            y_label = "Negative RMSE"
            
        fig = go.Figure(data=[
            go.Bar(x=models, y=scores, text=[f"{s:.3f}" for s in scores], textposition='auto')
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Models",
            yaxis_title=y_label,
            showlegend=False
        )
        
        return fig
    
    def _create_confusion_matrix(self, y_true, y_pred, model_name: str) -> go.Figure:
        """Create confusion matrix heatmap"""
        cm = confusion_matrix(y_true, y_pred)
        labels = sorted(y_true.unique())
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=[f"Predicted {label}" for label in labels],
            y=[f"Actual {label}" for label in labels],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title=f"Confusion Matrix - {model_name}",
            xaxis_title="Predicted",
            yaxis_title="Actual"
        )
        
        return fig
    
    def _create_roc_curve(self, y_true, y_proba, model_name: str) -> go.Figure:
        """Create ROC curve for binary classification"""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {roc_auc:.3f})',
            line=dict(width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', width=1)
        ))
        
        fig.update_layout(
            title=f"ROC Curve - {model_name}",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            showlegend=True
        )
        
        return fig
    
    def _create_residual_plots(self, y_true, y_pred, model_name: str) -> go.Figure:
        """Create residual plots for regression"""
        residuals = y_true - y_pred
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Residuals vs Predicted", "Residual Distribution")
        )
        
        # Residuals vs Predicted
        fig.add_trace(
            go.Scatter(
                x=y_pred, y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(size=6, opacity=0.6)
            ),
            row=1, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # Residual histogram
        fig.add_trace(
            go.Histogram(
                x=residuals,
                name='Residual Distribution',
                nbinsx=20
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f"Residual Analysis - {model_name}",
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Predicted Values", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=1, col=1)
        fig.update_xaxes(title_text="Residuals", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        
        return fig
    
    def _get_preprocessing_info(self) -> Dict[str, Any]:
        """Get information about preprocessing steps"""
        if not self.preprocessor:
            return {}
            
        info = {
            'steps': [],
            'transformers': []
        }
        
        # Extract transformer information
        for name, transformer, columns in self.preprocessor.transformers_:
            if name != 'remainder':
                info['transformers'].append({
                    'name': name,
                    'columns': len(columns) if hasattr(columns, '__len__') else 0,
                    'type': type(transformer).__name__
                })
        
        return info