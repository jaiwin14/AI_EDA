import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
import logging

# Scikit-learn
from sklearn.inspection import permutation_importance
from sklearn.base import is_classifier

# SHAP (with error handling)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

# Visualization
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class ExplainAgent:
    """Agent for model explainability using permutation importance and SHAP"""
    
    def __init__(self):
        self.name = "Explainability Agent"
        
    def process(self, context, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate model explanations
        
        Args:
            context: DatasetContext 
            baseline_results: Results from BaselineAgent
            
        Returns:
            Dict with feature importance and explanations
        """
        if baseline_results.get('status') != 'success':
            return {
                'status': 'error',
                'error': 'Baseline models not available',
                'feature_importance': {},
                'explanations': {}
            }
        
        best_model_name = baseline_results.get('best_model')
        if not best_model_name:
            return {
                'status': 'error',
                'error': 'No best model found',
                'feature_importance': {},
                'explanations': {}
            }
        
        try:
            # Get the best trained model
            trained_models = baseline_results.get('trained_pipelines', {})
            if best_model_name not in trained_models:
                return {
                    'status': 'error',
                    'error': f'Best model {best_model_name} not found in trained models',
                    'feature_importance': {},
                    'explanations': {}
                }
            
            best_model_info = trained_models[best_model_name]
            pipeline = best_model_info['pipeline']
            
            # Prepare data
            df = context.df
            target_col = context.target_column
            task_type = context.task_type
            
            X, y = self._prepare_data(df, target_col)
            
            # Calculate permutation importance
            perm_importance = self._calculate_permutation_importance(
                pipeline, X, y, task_type
            )
            
            # Calculate SHAP values (if available and applicable)
            shap_results = self._calculate_shap_values(
                pipeline, X, y, task_type, best_model_name
            )
            
            # Create visualizations
            visualizations = self._create_visualizations(
                perm_importance, shap_results, best_model_name
            )
            
            # Generate insights
            insights = self._generate_insights(
                perm_importance, shap_results, X.columns.tolist()
            )
            
            results = {
                'status': 'success',
                'best_model': best_model_name,
                'task_type': task_type,
                'feature_importance': {
                    'permutation': perm_importance,
                    'shap': shap_results
                },
                'visualizations': visualizations,
                'insights': insights,
                'methods_used': []
            }
            
            # Track methods used
            if perm_importance:
                results['methods_used'].append('permutation_importance')
            if shap_results:
                results['methods_used'].append('shap')
                
            return results
            
        except Exception as e:
            logger.error(f"Explain agent failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'feature_importance': {},
                'explanations': {}
            }
    
    def _prepare_data(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for explanation"""
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Remove rows with missing target
        mask = ~y.isnull()
        X = X[mask]
        y = y[mask]
        
        return X, y
    
    def _calculate_permutation_importance(self, pipeline, X: pd.DataFrame, 
                                        y: pd.Series, task_type: str) -> Dict[str, Any]:
        """Calculate permutation importance"""
        try:
            logger.info("Calculating permutation importance...")
            
            # Determine scoring method
            if task_type == 'classification':
                scoring = 'accuracy' if len(y.unique()) > 2 else 'roc_auc'
                try:
                    # Try with the preferred scoring
                    perm_imp = permutation_importance(
                        pipeline, X, y, n_repeats=5, random_state=42, 
                        scoring=scoring, n_jobs=-1
                    )
                except:
                    # Fallback to accuracy
                    perm_imp = permutation_importance(
                        pipeline, X, y, n_repeats=5, random_state=42, 
                        scoring='accuracy', n_jobs=-1
                    )
            else:
                perm_imp = permutation_importance(
                    pipeline, X, y, n_repeats=5, random_state=42, 
                    scoring='neg_mean_squared_error', n_jobs=-1
                )
            
            # Get feature names after preprocessing
            feature_names = self._get_feature_names(pipeline, X)
            
            # Create results dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance_mean': perm_imp.importances_mean,
                'importance_std': perm_imp.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            # Convert to dict format
            results = {
                'importances': importance_df.to_dict('records'),
                'feature_names': feature_names,
                'scoring_method': scoring if task_type == 'classification' else 'neg_mse',
                'n_repeats': 5
            }
            
            logger.info("✅ Permutation importance calculated successfully")
            return results
            
        except Exception as e:
            logger.error(f"Permutation importance calculation failed: {str(e)}")
            return {}
    
    def _calculate_shap_values(self, pipeline, X: pd.DataFrame, y: pd.Series, 
                             task_type: str, model_name: str) -> Dict[str, Any]:
        """Calculate SHAP values if applicable"""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available - skipping SHAP analysis")
            return {}
        
        try:
            # Extract the actual model from pipeline
            model = pipeline.named_steps['model']
            
            # Check if model is tree-based (best SHAP support)
            tree_models = ['DecisionTree', 'RandomForest', 'ExtraTree', 'GradientBoosting', 'XGBoost', 'LightGBM']
            is_tree_model = any(tree_name in type(model).__name__ for tree_name in tree_models)
            
            if not is_tree_model:
                logger.info(f"SHAP analysis skipped - {model_name} is not tree-based")
                return {}
            
            logger.info("Calculating SHAP values for tree model...")
            
            # Transform data through preprocessing
            X_transformed = pipeline.named_steps['preprocessor'].transform(X)
            
            # Convert to DataFrame with proper column names
            feature_names = self._get_feature_names(pipeline, X)
            if hasattr(X_transformed, 'toarray'):  # Handle sparse matrices
                X_transformed = X_transformed.toarray()
            X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
            
            # Sample data if too large (SHAP can be slow)
            if len(X_transformed_df) > 1000:
                sample_idx = np.random.choice(len(X_transformed_df), 1000, replace=False)
                X_sample = X_transformed_df.iloc[sample_idx]
            else:
                X_sample = X_transformed_df
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # Handle multiclass case
            if task_type == 'classification' and len(y.unique()) > 2:
                # For multiclass, take SHAP values for the first class
                if isinstance(shap_values, list):
                    shap_values_to_use = shap_values[0]
                else:
                    shap_values_to_use = shap_values
            else:
                # Binary classification or regression
                if isinstance(shap_values, list):
                    shap_values_to_use = shap_values[1] if len(shap_values) == 2 else shap_values[0]
                else:
                    shap_values_to_use = shap_values
            
            # Calculate feature importance from SHAP values
            shap_importance = np.abs(shap_values_to_use).mean(0)
            
            # Create SHAP results
            shap_df = pd.DataFrame({
                'feature': feature_names,
                'shap_importance': shap_importance
            }).sort_values('shap_importance', ascending=False)
            
            results = {
                'shap_values': shap_values_to_use.tolist(),
                'feature_importance': shap_df.to_dict('records'),
                'feature_names': feature_names,
                'base_value': explainer.expected_value,
                'sample_size': len(X_sample)
            }
            
            logger.info("✅ SHAP values calculated successfully")
            return results
            
        except Exception as e:
            logger.error(f"SHAP calculation failed: {str(e)}")
            return {}
    
    def _get_feature_names(self, pipeline, X_original: pd.DataFrame) -> List[str]:
        """Get feature names after preprocessing"""
        try:
            # Try to get feature names from preprocessor
            preprocessor = pipeline.named_steps['preprocessor']
            
            if hasattr(preprocessor, 'get_feature_names_out'):
                return list(preprocessor.get_feature_names_out())
            elif hasattr(preprocessor, 'get_feature_names'):
                return list(preprocessor.get_feature_names())
            else:
                # Fallback to original feature names
                return list(X_original.columns)
                
        except Exception as e:
            logger.warning(f"Could not get feature names from preprocessor: {e}")
            return list(X_original.columns)
    
    def _create_visualizations(self, perm_importance: Dict, shap_results: Dict, 
                             model_name: str) -> Dict[str, Any]:
        """Create visualization plots"""
        visualizations = {}
        
        # Permutation importance plot
        if perm_importance:
            try:
                importances = perm_importance['importances']
                if importances:
                    # Take top 15 features
                    top_features = importances[:15]
                    
                    features = [item['feature'] for item in top_features]
                    importance_mean = [item['importance_mean'] for item in top_features]
                    importance_std = [item['importance_std'] for item in top_features]
                    
                    # Create plotly figure
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        y=features[::-1],  # Reverse for horizontal bar
                        x=importance_mean[::-1],
                        error_x=dict(array=importance_std[::-1]),
                        orientation='h',
                        name='Permutation Importance',
                        marker_color='lightblue'
                    ))
                    
                    fig.update_layout(
                        title=f'Permutation Importance - {model_name}',
                        xaxis_title='Importance Score',
                        yaxis_title='Features',
                        height=max(400, len(features) * 30),
                        showlegend=False
                    )
                    
                    visualizations['permutation_importance'] = fig.to_dict()
                    
            except Exception as e:
                logger.error(f"Failed to create permutation importance plot: {e}")
        
        # SHAP importance plot
        if shap_results:
            try:
                shap_importance = shap_results['feature_importance']
                if shap_importance:
                    # Take top 15 features
                    top_features = shap_importance[:15]
                    
                    features = [item['feature'] for item in top_features]
                    importance = [item['shap_importance'] for item in top_features]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        y=features[::-1],
                        x=importance[::-1],
                        orientation='h',
                        name='SHAP Importance',
                        marker_color='lightcoral'
                    ))
                    
                    fig.update_layout(
                        title=f'SHAP Feature Importance - {model_name}',
                        xaxis_title='Mean |SHAP Value|',
                        yaxis_title='Features',
                        height=max(400, len(features) * 30),
                        showlegend=False
                    )
                    
                    visualizations['shap_importance'] = fig.to_dict()
                    
            except Exception as e:
                logger.error(f"Failed to create SHAP importance plot: {e}")
        
        return visualizations
    
    def _generate_insights(self, perm_importance: Dict, shap_results: Dict, 
                          all_features: List[str]) -> Dict[str, Any]:
        """Generate insights from feature importance analysis"""
        insights = {
            'summary': {},
            'top_features': {},
            'methodology': {},
            'recommendations': []
        }
        
        # Summary insights
        total_features = len(all_features)
        insights['summary']['total_features'] = total_features
        
        methods_used = []
        if perm_importance:
            methods_used.append('Permutation Importance')
        if shap_results:
            methods_used.append('SHAP Values')
        
        insights['methodology']['methods_used'] = methods_used
        
        # Top features analysis
        if perm_importance and perm_importance.get('importances'):
            perm_top = perm_importance['importances'][:5]
            insights['top_features']['permutation'] = [
                {
                    'feature': item['feature'],
                    'importance': round(item['importance_mean'], 4),
                    'std': round(item['importance_std'], 4)
                }
                for item in perm_top
            ]
            
            # Count features with positive importance
            positive_features = sum(1 for item in perm_importance['importances'] 
                                  if item['importance_mean'] > 0)
            insights['summary']['features_with_positive_importance'] = positive_features
        
        if shap_results and shap_results.get('feature_importance'):
            shap_top = shap_results['feature_importance'][:5]
            insights['top_features']['shap'] = [
                {
                    'feature': item['feature'],
                    'importance': round(item['shap_importance'], 4)
                }
                for item in shap_top
            ]
        
        # Generate recommendations
        if perm_importance:
            zero_importance_features = [
                item['feature'] for item in perm_importance['importances']
                if item['importance_mean'] <= 0
            ]
            
            if zero_importance_features:
                insights['recommendations'].append(
                    f"Consider removing {len(zero_importance_features)} features with zero or negative importance: {', '.join(zero_importance_features[:3])}{'...' if len(zero_importance_features) > 3 else ''}"
                )
            
            # Check for highly important features
            if perm_importance['importances']:
                top_importance = perm_importance['importances'][0]['importance_mean']
                if top_importance > 0.1:  # Significant importance
                    insights['recommendations'].append(
                        f"The feature '{perm_importance['importances'][0]['feature']}' shows very high importance ({top_importance:.3f}). Consider feature engineering around this variable."
                    )
        
        # Methodology insights
        if perm_importance:
            insights['methodology']['permutation_details'] = {
                'scoring_method': perm_importance.get('scoring_method'),
                'n_repeats': perm_importance.get('n_repeats')
            }
        
        if shap_results:
            insights['methodology']['shap_details'] = {
                'sample_size': shap_results.get('sample_size'),
                'base_value': shap_results.get('base_value')
            }
        
        return insights
    
    def create_summary_report(self, results: Dict[str, Any]) -> str:
        """Create a text summary report of explainability results"""
        if results.get('status') != 'success':
            return f"❌ Explainability analysis failed: {results.get('error', 'Unknown error')}"
        
        report = []
        report.append("🔍 MODEL EXPLAINABILITY REPORT")
        report.append("=" * 50)
        
        # Basic info
        model_name = results.get('best_model', 'Unknown')
        task_type = results.get('task_type', 'Unknown')
        methods = results.get('methods_used', [])
        
        report.append(f"📊 Model: {model_name}")
        report.append(f"📈 Task Type: {task_type}")
        report.append(f"🔧 Methods Used: {', '.join(methods)}")
        report.append("")
        
        # Insights summary
        insights = results.get('insights', {})
        summary = insights.get('summary', {})
        
        if summary:
            report.append("📋 SUMMARY")
            report.append("-" * 20)
            report.append(f"Total Features: {summary.get('total_features', 'N/A')}")
            
            if 'features_with_positive_importance' in summary:
                pos_features = summary['features_with_positive_importance']
                total = summary.get('total_features', 1)
                report.append(f"Features with Positive Importance: {pos_features}/{total} ({pos_features/total*100:.1f}%)")
            report.append("")
        
        # Top features
        top_features = insights.get('top_features', {})
        
        if 'permutation' in top_features:
            report.append("🏆 TOP FEATURES (Permutation Importance)")
            report.append("-" * 40)
            for i, feat in enumerate(top_features['permutation'][:5], 1):
                report.append(f"{i}. {feat['feature']}: {feat['importance']:.4f} (±{feat['std']:.4f})")
            report.append("")
        
        if 'shap' in top_features:
            report.append("🏆 TOP FEATURES (SHAP Importance)")
            report.append("-" * 35)
            for i, feat in enumerate(top_features['shap'][:5], 1):
                report.append(f"{i}. {feat['feature']}: {feat['importance']:.4f}")
            report.append("")
        
        # Recommendations
        recommendations = insights.get('recommendations', [])
        if recommendations:
            report.append("💡 RECOMMENDATIONS")
            report.append("-" * 20)
            for i, rec in enumerate(recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        # Methodology
        methodology = insights.get('methodology', {})
        if methodology:
            report.append("🔬 METHODOLOGY")
            report.append("-" * 15)
            
            if 'permutation_details' in methodology:
                perm_details = methodology['permutation_details']
                report.append(f"Permutation Importance: {perm_details.get('scoring_method')} with {perm_details.get('n_repeats')} repeats")
            
            if 'shap_details' in methodology:
                shap_details = methodology['shap_details']
                report.append(f"SHAP Analysis: {shap_details.get('sample_size')} samples, base value: {shap_details.get('base_value'):.4f}")
        
        return "\n".join(report)