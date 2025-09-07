"""
Gemini API Integration for Narrative Generation
Generate human-readable narratives for EDA results and model performance
"""

import logging
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from app.core.config import settings

logger = logging.getLogger(__name__)

class GeminiNarrator:
    """Generate narratives using Google's Gemini API"""
    
    def __init__(self):
        self.model = None
        self.initialized = False
        
    async def initialize(self) -> Dict[str, Any]:
        """Initialize Gemini API client"""
        
        if not GEMINI_AVAILABLE:
            return {
                "status": "error",
                "message": "Gemini API library not available. Install with: pip install google-generativeai"
            }
        
        if not settings.GEMINI_API_KEY:
            return {
                "status": "error",
                "message": "Gemini API key not configured. Set GEMINI_API_KEY environment variable."
            }
        
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
            self.initialized = True
            
            return {
                "status": "success",
                "message": "Gemini API initialized successfully",
                "model": settings.GEMINI_MODEL
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to initialize Gemini API: {str(e)}"
            }
    
    async def generate_eda_narrative(self, eda_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate narrative summary of EDA results
        
        Args:
            eda_results: Dictionary containing EDA analysis results
        """
        
        if not self.initialized:
            init_result = await self.initialize()
            if init_result["status"] != "success":
                return init_result
        
        try:
            # Create structured prompt for EDA narrative
            prompt = self._create_eda_prompt(eda_results)
            
            # Generate narrative
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=settings.MAX_TOKENS,
                    temperature=0.3
                )
            )
            
            narrative = response.text
            
            return {
                "status": "success",
                "narrative": narrative,
                "narrative_type": "eda_summary",
                "generated_at": datetime.utcnow().isoformat(),
                "token_count": len(narrative.split())
            }
            
        except Exception as e:
            logger.error(f"Failed to generate EDA narrative: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to generate narrative: {str(e)}"
            }
    
    async def generate_model_narrative(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate narrative summary of model training results
        
        Args:
            model_results: Dictionary containing model training results
        """
        
        if not self.initialized:
            init_result = await self.initialize()
            if init_result["status"] != "success":
                return init_result
        
        try:
            # Create structured prompt for model narrative
            prompt = self._create_model_prompt(model_results)
            
            # Generate narrative
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=settings.MAX_TOKENS,
                    temperature=0.3
                )
            )
            
            narrative = response.text
            
            return {
                "status": "success",
                "narrative": narrative,
                "narrative_type": "model_summary",
                "generated_at": datetime.utcnow().isoformat(),
                "token_count": len(narrative.split())
            }
            
        except Exception as e:
            logger.error(f"Failed to generate model narrative: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to generate narrative: {str(e)}"
            }
    
    async def generate_prediction_narrative(self, prediction_data: Dict[str, Any], 
                                          explanation_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate narrative explanation of predictions
        
        Args:
            prediction_data: Dictionary containing prediction results
            explanation_data: Optional SHAP explanation data
        """
        
        if not self.initialized:
            init_result = await self.initialize()
            if init_result["status"] != "success":
                return init_result
        
        try:
            # Create structured prompt for prediction narrative
            prompt = self._create_prediction_prompt(prediction_data, explanation_data)
            
            # Generate narrative
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=settings.MAX_TOKENS,
                    temperature=0.4
                )
            )
            
            narrative = response.text
            
            return {
                "status": "success",
                "narrative": narrative,
                "narrative_type": "prediction_explanation",
                "generated_at": datetime.utcnow().isoformat(),
                "token_count": len(narrative.split())
            }
            
        except Exception as e:
            logger.error(f"Failed to generate prediction narrative: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to generate narrative: {str(e)}"
            }
    
    def _create_eda_prompt(self, eda_results: Dict[str, Any]) -> str:
        """Create structured prompt for EDA narrative generation"""
        
        prompt = """
You are a data science expert tasked with creating a comprehensive, human-readable summary of exploratory data analysis results. 

Based on the following EDA results, generate a clear, insightful narrative that explains:
1. Dataset overview and key characteristics
2. Data quality issues and recommendations
3. Important patterns and relationships discovered
4. Statistical insights and their business implications
5. Actionable recommendations for data preprocessing and modeling

EDA Results:
"""
        
        # Add dataset overview
        if "analyses" in eda_results:
            analyses = eda_results["analyses"]
            
            # Basic statistics
            if "basic_statistics" in analyses:
                basic_stats = analyses["basic_statistics"]
                prompt += f"\n**Dataset Overview:**\n"
                if "overview" in basic_stats:
                    overview = basic_stats["overview"]
                    prompt += f"- Dataset contains {overview.get('total_rows', 'N/A')} rows and {overview.get('total_columns', 'N/A')} columns\n"
                    prompt += f"- {overview.get('numeric_columns', 0)} numeric columns, {overview.get('categorical_columns', 0)} categorical columns\n"
                    prompt += f"- Memory usage: {overview.get('memory_usage_mb', 0):.1f} MB\n"
                    if overview.get('duplicate_rows', 0) > 0:
                        prompt += f"- Found {overview.get('duplicate_rows', 0)} duplicate rows\n"
            
            # Missing values
            if "missing_values" in analyses:
                missing_analysis = analyses["missing_values"]
                prompt += f"\n**Missing Values:**\n"
                prompt += f"- Overall missing percentage: {missing_analysis.get('overall_missing_percentage', 0):.1f}%\n"
                
                columns_with_missing = missing_analysis.get('columns_with_missing', {})
                if columns_with_missing:
                    prompt += f"- Columns with missing values: {list(columns_with_missing.keys())}\n"
                
                recommendations = missing_analysis.get('recommendations', [])
                if recommendations:
                    prompt += f"- Recommendations: {[rec.get('suggestion', '') for rec in recommendations]}\n"
            
            # Correlations
            if "correlations" in analyses:
                corr_analysis = analyses["correlations"]
                strong_corrs = corr_analysis.get('strong_correlations', {}).get('pearson', [])
                if strong_corrs:
                    prompt += f"\n**Strong Correlations Found:**\n"
                    for corr in strong_corrs[:5]:  # Top 5
                        prompt += f"- {corr.get('feature_1')} vs {corr.get('feature_2')}: {corr.get('correlation', 0):.3f}\n"
            
            # Outliers
            if "outliers" in analyses:
                outlier_analysis = analyses["outliers"]
                outlier_summary = outlier_analysis.get('outlier_summary', {})
                high_outlier_cols = [col for col, info in outlier_summary.items() 
                                   if info.get('severity') == 'high']
                if high_outlier_cols:
                    prompt += f"\n**Outlier Issues:**\n"
                    prompt += f"- Columns with high outlier concentration: {high_outlier_cols}\n"
        
        prompt += """

Please provide a comprehensive narrative that:
- Uses clear, business-friendly language
- Highlights the most important findings
- Provides actionable insights and recommendations
- Explains potential impacts on machine learning model performance
- Suggests specific preprocessing steps

Format the response with clear sections and bullet points where appropriate.
"""
        
        return prompt
    
    def _create_model_prompt(self, model_results: Dict[str, Any]) -> str:
        """Create structured prompt for model narrative generation"""
        
        prompt = """
You are a machine learning expert tasked with explaining model training results in clear, business-friendly language.

Based on the following model training results, generate a comprehensive narrative that explains:
1. Task type and target variable analysis
2. Model performance comparison and recommendations
3. Best performing model and its characteristics
4. Business implications of the results
5. Recommendations for model deployment and monitoring

Model Training Results:
"""
        
        # Add task information
        task_type = model_results.get('task_type', 'Unknown')
        target_column = model_results.get('target_column', 'Unknown')
        prompt += f"\n**Task Information:**\n"
        prompt += f"- Task Type: {task_type.title()}\n"
        prompt += f"- Target Variable: {target_column}\n"
        
        # Add dataset info
        dataset_info = model_results.get('dataset_info', {})
        if dataset_info:
            prompt += f"- Training samples: {dataset_info.get('train_samples', 'N/A')}\n"
            prompt += f"- Test samples: {dataset_info.get('test_samples', 'N/A')}\n"
            prompt += f"- Number of features: {dataset_info.get('n_features', 'N/A')}\n"
        
        # Add model results
        models = model_results.get('models', {})
        best_model = model_results.get('best_model')
        
        prompt += f"\n**Model Performance:**\n"
        for model_name, model_data in models.items():
            if model_data.get('status') == 'completed':
                metrics = model_data.get('metrics', {})
                test_metrics = metrics.get('test', {})
                
                if task_type == 'classification':
                    accuracy = test_metrics.get('accuracy', 0)
                    f1_score = test_metrics.get('f1_score', 0)
                    prompt += f"- {model_name}: Accuracy={accuracy:.3f}, F1-Score={f1_score:.3f}\n"
                else:
                    rmse = test_metrics.get('rmse', 0)
                    r2 = test_metrics.get('r2', 0)
                    prompt += f"- {model_name}: RMSE={rmse:.3f}, R²={r2:.3f}\n"
        
        if best_model:
            prompt += f"\n**Best Model:** {best_model}\n"
        
        # Add model comparison
        model_comparison = model_results.get('model_comparison', {})
        if model_comparison:
            prompt += f"\n**Model Comparison Summary:**\n{json.dumps(model_comparison, indent=2)}\n"
        
        prompt += """

Please provide a comprehensive narrative that:
- Explains the results in business-friendly terms
- Compares model performance and recommends the best approach
- Discusses the reliability and expected performance of the models
- Provides guidance on model deployment considerations
- Suggests monitoring strategies and potential improvements
- Explains what the metrics mean in practical terms

Use clear language suitable for both technical and non-technical stakeholders.
"""
        
        return prompt
    
    def _create_prediction_prompt(self, prediction_data: Dict[str, Any], 
                                explanation_data: Optional[Dict[str, Any]] = None) -> str:
        """Create structured prompt for prediction narrative generation"""
        
        prompt = """
You are an AI assistant explaining machine learning predictions to users in clear, understandable language.

Based on the following prediction results and explanations, generate a narrative that:
1. Clearly states the prediction and confidence level
2. Explains the key factors that influenced the prediction
3. Provides context about the reliability of the prediction
4. Offers actionable insights based on the results

Prediction Data:
"""
        
        # Add prediction information
        prediction = prediction_data.get('prediction')
        probabilities = prediction_data.get('probabilities')
        input_data = prediction_data.get('input_data', {})
        
        prompt += f"\n**Prediction:** {prediction}\n"
        
        if probabilities:
            prompt += f"**Confidence/Probabilities:** {probabilities}\n"
        
        prompt += f"**Input Features:** {json.dumps(input_data, indent=2)}\n"
        
        # Add explanation data if available
        if explanation_data and explanation_data.get('status') == 'success':
            explanations = explanation_data.get('explanations', [])
            if explanations:
                first_explanation = explanations[0]
                shap_values = first_explanation.get('shap_values', {})
                
                # Sort SHAP values by absolute impact
                sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
                
                prompt += f"\n**Feature Importance (SHAP values):**\n"
                for feature, impact in sorted_shap[:10]:  # Top 10 features
                    direction = "increases" if impact > 0 else "decreases"
                    prompt += f"- {feature}: {impact:.4f} ({direction} prediction)\n"
        
        prompt += """

Please provide a clear, conversational explanation that:
- States the prediction in simple terms
- Explains what factors were most important in making this prediction
- Discusses the confidence level and what it means
- Provides practical insights about the prediction
- Uses analogies or examples where helpful
- Avoids technical jargon while remaining accurate

Keep the explanation concise but informative, suitable for a general audience.
"""
        
        return prompt
