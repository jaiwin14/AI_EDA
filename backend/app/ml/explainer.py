"""
SHAP-based Model Explainability
Generate explanations for model predictions using SHAP
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union
import joblib
from pathlib import Path

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from app.core.config import settings

logger = logging.getLogger(__name__)

class ModelExplainer:
    """SHAP-based model explainer for generating interpretable explanations"""
    
    def __init__(self):
        self.explainer = None
        self.model = None
        self.feature_names = []
        
    async def initialize_explainer(self, model_path: str, background_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Initialize SHAP explainer for a trained model
        
        Args:
            model_path: Path to the trained model
            background_data: Background dataset for SHAP explainer (optional)
        """
        
        if not SHAP_AVAILABLE:
            return {
                "status": "error",
                "message": "SHAP library not available. Install with: pip install shap"
            }
        
        try:
            # Load model
            self.model = joblib.load(model_path)
            
            # Extract feature names from the pipeline
            if hasattr(self.model, 'named_steps'):
                preprocessor = self.model.named_steps.get('preprocessor')
                if preprocessor and hasattr(preprocessor, 'get_feature_names_out'):
                    try:
                        self.feature_names = preprocessor.get_feature_names_out().tolist()
                    except:
                        # Fallback for older sklearn versions
                        self.feature_names = [f"feature_{i}" for i in range(background_data.shape[1] if background_data is not None else 10)]
            
            # Initialize SHAP explainer
            if background_data is not None:
                # Use background data for explainer
                background_processed = self.model.named_steps['preprocessor'].transform(background_data)
                
                # Choose appropriate explainer based on model type
                model_step = self.model.named_steps['model']
                
                if hasattr(model_step, 'predict_proba'):
                    # For classification models
                    self.explainer = shap.Explainer(
                        self.model.predict_proba,
                        background_processed,
                        feature_names=self.feature_names
                    )
                else:
                    # For regression models
                    self.explainer = shap.Explainer(
                        self.model.predict,
                        background_processed,
                        feature_names=self.feature_names
                    )
            else:
                # Use model-specific explainer without background data
                model_step = self.model.named_steps['model']
                
                if str(type(model_step)).find('RandomForest') != -1:
                    self.explainer = shap.TreeExplainer(model_step)
                elif str(type(model_step)).find('LinearRegression') != -1 or str(type(model_step)).find('LogisticRegression') != -1:
                    self.explainer = shap.LinearExplainer(model_step, background_processed if background_data is not None else None)
                else:
                    # Generic explainer
                    self.explainer = shap.Explainer(self.model.predict)
            
            return {
                "status": "success",
                "message": "SHAP explainer initialized successfully",
                "explainer_type": str(type(self.explainer).__name__)
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to initialize explainer: {str(e)}"
            }
    
    async def explain_prediction(self, input_data: pd.DataFrame, 
                               explanation_type: str = "local") -> Dict[str, Any]:
        """
        Generate SHAP explanations for predictions
        
        Args:
            input_data: Input data for explanation
            explanation_type: Type of explanation ('local', 'global', 'summary')
        """
        
        if not SHAP_AVAILABLE:
            return {
                "status": "error",
                "message": "SHAP library not available"
            }
        
        if self.explainer is None:
            return {
                "status": "error",
                "message": "Explainer not initialized. Call initialize_explainer first."
            }
        
        try:
            # Preprocess input data
            if hasattr(self.model, 'named_steps'):
                processed_data = self.model.named_steps['preprocessor'].transform(input_data)
            else:
                processed_data = input_data
            
            # Generate SHAP values
            shap_values = self.explainer(processed_data)
            
            explanations = {
                "status": "success",
                "explanation_type": explanation_type,
                "predictions": self.model.predict(input_data).tolist(),
                "explanations": []
            }
            
            # Process explanations based on type
            if explanation_type == "local":
                # Individual prediction explanations
                for i in range(len(input_data)):
                    explanation = {
                        "instance_index": i,
                        "input_values": input_data.iloc[i].to_dict(),
                        "shap_values": {},
                        "base_value": float(shap_values.base_values[i]) if hasattr(shap_values, 'base_values') else 0.0,
                        "prediction": float(explanations["predictions"][i])
                    }
                    
                    # Extract SHAP values for features
                    if hasattr(shap_values, 'values'):
                        if len(shap_values.values.shape) == 3:  # Multi-class classification
                            # Take the first class for simplicity
                            feature_impacts = shap_values.values[i, :, 0]
                        else:
                            feature_impacts = shap_values.values[i]
                        
                        # Map to feature names
                        for j, impact in enumerate(feature_impacts):
                            feature_name = self.feature_names[j] if j < len(self.feature_names) else f"feature_{j}"
                            explanation["shap_values"][feature_name] = float(impact)
                    
                    explanations["explanations"].append(explanation)
            
            elif explanation_type == "global":
                # Global feature importance
                if hasattr(shap_values, 'values'):
                    if len(shap_values.values.shape) == 3:  # Multi-class
                        mean_shap = np.abs(shap_values.values[:, :, 0]).mean(axis=0)
                    else:
                        mean_shap = np.abs(shap_values.values).mean(axis=0)
                    
                    feature_importance = {}
                    for j, importance in enumerate(mean_shap):
                        feature_name = self.feature_names[j] if j < len(self.feature_names) else f"feature_{j}"
                        feature_importance[feature_name] = float(importance)
                    
                    # Sort by importance
                    sorted_importance = dict(sorted(feature_importance.items(), 
                                                  key=lambda x: x[1], reverse=True))
                    
                    explanations["global_importance"] = sorted_importance
            
            elif explanation_type == "summary":
                # Summary statistics
                if hasattr(shap_values, 'values'):
                    if len(shap_values.values.shape) == 3:  # Multi-class
                        values_to_analyze = shap_values.values[:, :, 0]
                    else:
                        values_to_analyze = shap_values.values
                    
                    summary_stats = {
                        "mean_abs_shap": np.abs(values_to_analyze).mean(axis=0).tolist(),
                        "std_shap": values_to_analyze.std(axis=0).tolist(),
                        "max_shap": values_to_analyze.max(axis=0).tolist(),
                        "min_shap": values_to_analyze.min(axis=0).tolist()
                    }
                    
                    explanations["summary_statistics"] = summary_stats
            
            return explanations
            
        except Exception as e:
            logger.error(f"Failed to generate SHAP explanation: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to generate explanation: {str(e)}"
            }
    
    async def generate_explanation_plots(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate SHAP visualization plots
        
        Args:
            input_data: Input data for visualization
        """
        
        if not SHAP_AVAILABLE:
            return {
                "status": "error",
                "message": "SHAP library not available"
            }
        
        if self.explainer is None:
            return {
                "status": "error",
                "message": "Explainer not initialized"
            }
        
        try:
            # Preprocess input data
            if hasattr(self.model, 'named_steps'):
                processed_data = self.model.named_steps['preprocessor'].transform(input_data)
            else:
                processed_data = input_data
            
            # Generate SHAP values
            shap_values = self.explainer(processed_data)
            
            plots = {
                "status": "success",
                "available_plots": []
            }
            
            # Generate different types of plots
            try:
                # Waterfall plot for first instance
                if len(input_data) > 0:
                    waterfall_data = self._create_waterfall_data(shap_values, 0)
                    plots["waterfall_plot"] = waterfall_data
                    plots["available_plots"].append("waterfall")
            except Exception as e:
                logger.warning(f"Failed to create waterfall plot: {str(e)}")
            
            try:
                # Summary plot data
                summary_data = self._create_summary_plot_data(shap_values, input_data)
                plots["summary_plot"] = summary_data
                plots["available_plots"].append("summary")
            except Exception as e:
                logger.warning(f"Failed to create summary plot: {str(e)}")
            
            try:
                # Feature importance plot
                importance_data = self._create_importance_plot_data(shap_values)
                plots["importance_plot"] = importance_data
                plots["available_plots"].append("importance")
            except Exception as e:
                logger.warning(f"Failed to create importance plot: {str(e)}")
            
            return plots
            
        except Exception as e:
            logger.error(f"Failed to generate SHAP plots: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to generate plots: {str(e)}"
            }
    
    def _create_waterfall_data(self, shap_values, instance_index: int) -> Dict[str, Any]:
        """Create data for waterfall plot"""
        
        if not hasattr(shap_values, 'values'):
            return {"error": "No SHAP values available"}
        
        # Get SHAP values for the instance
        if len(shap_values.values.shape) == 3:  # Multi-class
            instance_shap = shap_values.values[instance_index, :, 0]
        else:
            instance_shap = shap_values.values[instance_index]
        
        base_value = float(shap_values.base_values[instance_index]) if hasattr(shap_values, 'base_values') else 0.0
        
        # Create waterfall data
        waterfall_data = {
            "base_value": base_value,
            "features": [],
            "shap_values": [],
            "feature_values": [],
            "cumulative_values": []
        }
        
        cumulative = base_value
        
        # Sort features by absolute SHAP value
        feature_indices = np.argsort(np.abs(instance_shap))[::-1]
        
        for i in feature_indices:
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            shap_val = float(instance_shap[i])
            
            waterfall_data["features"].append(feature_name)
            waterfall_data["shap_values"].append(shap_val)
            waterfall_data["feature_values"].append("N/A")  # Would need original feature values
            
            cumulative += shap_val
            waterfall_data["cumulative_values"].append(cumulative)
        
        return waterfall_data
    
    def _create_summary_plot_data(self, shap_values, input_data: pd.DataFrame) -> Dict[str, Any]:
        """Create data for summary plot"""
        
        if not hasattr(shap_values, 'values'):
            return {"error": "No SHAP values available"}
        
        # Get SHAP values
        if len(shap_values.values.shape) == 3:  # Multi-class
            values_to_plot = shap_values.values[:, :, 0]
        else:
            values_to_plot = shap_values.values
        
        # Calculate feature importance (mean absolute SHAP value)
        feature_importance = np.abs(values_to_plot).mean(axis=0)
        
        # Sort features by importance
        feature_indices = np.argsort(feature_importance)[::-1]
        
        summary_data = {
            "features": [],
            "importance_scores": [],
            "shap_distributions": []
        }
        
        for i in feature_indices[:20]:  # Top 20 features
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            importance = float(feature_importance[i])
            shap_dist = values_to_plot[:, i].tolist()
            
            summary_data["features"].append(feature_name)
            summary_data["importance_scores"].append(importance)
            summary_data["shap_distributions"].append(shap_dist)
        
        return summary_data
    
    def _create_importance_plot_data(self, shap_values) -> Dict[str, Any]:
        """Create data for feature importance plot"""
        
        if not hasattr(shap_values, 'values'):
            return {"error": "No SHAP values available"}
        
        # Get SHAP values
        if len(shap_values.values.shape) == 3:  # Multi-class
            values_to_analyze = shap_values.values[:, :, 0]
        else:
            values_to_analyze = shap_values.values
        
        # Calculate mean absolute SHAP values
        importance_scores = np.abs(values_to_analyze).mean(axis=0)
        
        # Create importance data
        importance_data = {
            "features": [],
            "importance_scores": []
        }
        
        # Sort by importance
        feature_indices = np.argsort(importance_scores)[::-1]
        
        for i in feature_indices:
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            importance = float(importance_scores[i])
            
            importance_data["features"].append(feature_name)
            importance_data["importance_scores"].append(importance)
        
        return importance_data
