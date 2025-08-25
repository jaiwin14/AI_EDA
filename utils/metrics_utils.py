import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import warnings
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error,
    classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_proba: Optional[np.ndarray] = None,
                                   labels: Optional[List] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (for probabilistic metrics)
        labels: Class labels
        
    Returns:
        Dictionary with all classification metrics
    """
    metrics = {}
    
    try:
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # Determine if binary or multiclass
        unique_labels = np.unique(y_true)
        is_binary = len(unique_labels) == 2
        
        if is_binary:
            # Binary classification metrics
            metrics['precision'] = precision_score(y_true, y_pred)
            metrics['recall'] = recall_score(y_true, y_pred)
            metrics['f1'] = f1_score(y_true, y_pred)
            metrics['specificity'] = calculate_specificity(y_true, y_pred)
            
            # Probabilistic metrics (if probabilities available)
            if y_proba is not None:
                if y_proba.ndim == 2:
                    # Take positive class probabilities
                    y_proba_pos = y_proba[:, 1]
                else:
                    y_proba_pos = y_proba
                    
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba_pos)
                metrics['pr_auc'] = average_precision_score(y_true, y_proba_pos)
                metrics['log_loss'] = log_loss(y_true, y_proba if y_proba.ndim == 2 else 
                                             np.column_stack([1-y_proba, y_proba]))
        else:
            # Multiclass classification metrics
            metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
            metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
            metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
            metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
            
            # Multiclass probabilistic metrics
            if y_proba is not None:
                try:
                    metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
                    metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_proba, multi_class='ovo')
                    metrics['log_loss'] = log_loss(y_true, y_proba)
                except:
                    pass
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        # Classification report
        if labels:
            metrics['classification_report'] = classification_report(
                y_true, y_pred, target_names=labels, output_dict=True
            )
        else:
            metrics['classification_report'] = classification_report(
                y_true, y_pred, output_dict=True
            )
            
    except Exception as e:
        print(f"Error calculating classification metrics: {e}")
        metrics['error'] = str(e)
    
    return metrics

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with regression metrics
    """
    metrics = {}
    
    try:
        # Basic metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # MAPE with zero protection
        mask = y_true != 0
        if mask.sum() > 0:
            metrics['mape'] = mean_absolute_percentage_error(y_true[mask], y_pred[mask])
        else:
            metrics['mape'] = np.inf
            
        # Additional metrics
        metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
        metrics['max_error'] = np.max(np.abs(y_true - y_pred))
        metrics['mean_residual'] = np.mean(y_true - y_pred)
        metrics['std_residual'] = np.std(y_true - y_pred)
        
        # Adjusted R²
        n = len(y_true)
        p = 1  # Assuming simple regression, adjust if needed
        if n > p + 1:
            metrics['adjusted_r2'] = 1 - ((1 - metrics['r2']) * (n - 1) / (n - p - 1))
        else:
            metrics['adjusted_r2'] = metrics['r2']
            
    except Exception as e:
        print(f"Error calculating regression metrics: {e}")
        metrics['error'] = str(e)
    
    return metrics

def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate specificity (True Negative Rate)"""
    try:
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            return specificity
        else:
            return 0.0
    except:
        return 0.0

def explained_variance_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate explained variance score"""
    try:
        from sklearn.metrics import explained_variance_score as evs
        return evs(y_true, y_pred)
    except:
        try:
            # Manual calculation
            y_true_mean = np.mean(y_true)
            ss_tot = np.sum((y_true - y_true_mean) ** 2)
            ss_res = np.sum((y_true - y_pred) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        except:
            return 0.0

def create_metrics_summary_table(metrics: Dict[str, Any], task_type: str) -> pd.DataFrame:
    """
    Create a formatted table summarizing key metrics
    
    Args:
        metrics: Dictionary of calculated metrics
        task_type: 'classification' or 'regression'
        
    Returns:
        Formatted DataFrame with metrics
    """
    try:
        if task_type == 'classification':
            # Select key classification metrics
            key_metrics = {}
            
            # Always include these if available
            for metric in ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall', 
                          'f1_weighted', 'roc_auc', 'pr_auc']:
                if metric in metrics:
                    key_metrics[metric.replace('_', ' ').title()] = f"{metrics[metric]:.4f}"
            
            # Add log loss if available
            if 'log_loss' in metrics:
                key_metrics['Log Loss'] = f"{metrics['log_loss']:.4f}"
                
        else:  # regression
            key_metrics = {}
            
            # Always include these if available
            for metric in ['mae', 'rmse', 'r2', 'mape', 'explained_variance', 'adjusted_r2']:
                if metric in metrics:
                    value = metrics[metric]
                    if metric == 'mape' and np.isinf(value):
                        key_metrics[metric.replace('_', ' ').upper()] = "N/A"
                    else:
                        key_metrics[metric.replace('_', ' ').upper()] = f"{value:.4f}"
        
        # Convert to DataFrame
        df = pd.DataFrame(list(key_metrics.items()), columns=['Metric', 'Value'])
        return df
        
    except Exception as e:
        print(f"Error creating metrics table: {e}")
        return pd.DataFrame({'Metric': ['Error'], 'Value': [str(e)]})

def calculate_cross_validation_metrics(cv_scores: np.ndarray, scoring: str) -> Dict[str, float]:
    """
    Calculate summary statistics from cross-validation scores
    
    Args:
        cv_scores: Array of CV scores
        scoring: Scoring method used
        
    Returns:
        Dictionary with CV statistics
    """
    try:
        return {
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'cv_min': np.min(cv_scores),
            'cv_max': np.max(cv_scores),
            'cv_median': np.median(cv_scores),
            'scoring_method': scoring,
            'n_folds': len(cv_scores)
        }
    except:
        return {
            'cv_mean': 0.0,
            'cv_std': 0.0,
            'cv_min': 0.0,
            'cv_max': 0.0,
            'cv_median': 0.0,
            'scoring_method': scoring,
            'n_folds': 0
        }

def calculate_feature_importance_stats(importance_scores: np.ndarray, 
                                     feature_names: List[str]) -> Dict[str, Any]:
    """
    Calculate statistics from feature importance scores
    
    Args:
        importance_scores: Array of importance scores
        feature_names: List of feature names
        
    Returns:
        Dictionary with importance statistics
    """
    try:
        # Basic statistics
        stats = {
            'mean_importance': np.mean(importance_scores),
            'std_importance': np.std(importance_scores),
            'max_importance': np.max(importance_scores),
            'min_importance': np.min(importance_scores),
            'zero_importance_count': np.sum(importance_scores == 0),
            'zero_importance_ratio': np.sum(importance_scores == 0) / len(importance_scores)
        }
        
        # Top features
        sorted_indices = np.argsort(importance_scores)[::-1]
        stats['top_features'] = {
            'top_5_features': [feature_names[i] for i in sorted_indices[:5]],
            'top_5_scores': importance_scores[sorted_indices[:5]].tolist(),
            'top_10_features': [feature_names[i] for i in sorted_indices[:10]],
            'top_10_scores': importance_scores[sorted_indices[:10]].tolist()
        }
        
        # Distribution analysis
        stats['importance_quartiles'] = {
            'q25': np.percentile(importance_scores, 25),
            'q50': np.percentile(importance_scores, 50),
            'q75': np.percentile(importance_scores, 75)
        }
        
        return stats
        
    except Exception as e:
        print(f"Error calculating feature importance stats: {e}")
        return {'error': str(e)}

def compare_model_performance(models_metrics: Dict[str, Dict], 
                            primary_metric: str = 'accuracy') -> pd.DataFrame:
    """
    Compare performance across multiple models
    
    Args:
        models_metrics: Dictionary of model names to their metrics
        primary_metric: Primary metric for ranking
        
    Returns:
        DataFrame with model comparison
    """
    try:
        comparison_data = []
        
        for model_name, metrics in models_metrics.items():
            if 'error' not in metrics:
                row = {
                    'Model': model_name,
                    'Primary_Metric': metrics.get(primary_metric, 0),
                    'Status': 'Success'
                }
                
                # Add other relevant metrics
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        row[key] = value
                        
                comparison_data.append(row)
            else:
                comparison_data.append({
                    'Model': model_name,
                    'Primary_Metric': 0,
                    'Status': 'Failed',
                    'Error': metrics.get('error', 'Unknown error')
                })
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by primary metric (descending for most metrics, ascending for error metrics)
        ascending = primary_metric.lower() in ['mae', 'mse', 'rmse', 'log_loss']
        df = df.sort_values('Primary_Metric', ascending=ascending)
        
        return df
        
    except Exception as e:
        print(f"Error comparing model performance: {e}")
        return pd.DataFrame({'Error': [str(e)]})

def calculate_prediction_intervals(y_true: np.ndarray, y_pred: np.ndarray, 
                                 confidence: float = 0.95) -> Dict[str, np.ndarray]:
    """
    Calculate prediction intervals for regression
    
    Args:
        y_true: True values
        y_pred: Predicted values
        confidence: Confidence level (0-1)
        
    Returns:
        Dictionary with prediction intervals
    """
    try:
        residuals = y_true - y_pred
        residual_std = np.std(residuals)
        
        # Calculate z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        # Calculate intervals
        margin_of_error = z_score * residual_std
        lower_bound = y_pred - margin_of_error
        upper_bound = y_pred + margin_of_error
        
        # Calculate coverage (percentage of true values within intervals)
        within_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
        coverage = np.mean(within_interval)
        
        return {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'margin_of_error': margin_of_error,
            'coverage': coverage,
            'confidence_level': confidence
        }
        
    except Exception as e:
        print(f"Error calculating prediction intervals: {e}")
        return {'error': str(e)}

def format_metrics_for_display(metrics: Dict[str, Any], decimal_places: int = 4) -> Dict[str, str]:
    """
    Format metrics for display in UI
    
    Args:
        metrics: Raw metrics dictionary
        decimal_places: Number of decimal places
        
    Returns:
        Formatted metrics dictionary
    """
    formatted = {}
    
    try:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    formatted[key] = "N/A"
                else:
                    formatted[key] = f"{value:.{decimal_places}f}"
            elif isinstance(value, dict):
                # Recursively format nested dictionaries
                formatted[key] = format_metrics_for_display(value, decimal_places)
            else:
                formatted[key] = str(value)
                
    except Exception as e:
        formatted['format_error'] = str(e)
    
    return formatted