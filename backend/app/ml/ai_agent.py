"""
Centralized AI Agent with Multiple Prompt Templates
Handles all AI-powered analysis across different EDA steps
"""
import google.generativeai as genai
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from enum import Enum
import json
from datetime import datetime

from app.core.config import settings
# Assuming PromptTemplates class is defined elsewhere and imported.
# For this corrected code, I'll include a placeholder for it.
# from .prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Available analysis types"""
    DATASET_OVERVIEW = "dataset_overview"
    STATISTICAL_SUMMARY = "statistical_summary"
    CORRELATION_INSIGHTS = "correlation_insights"
    MISSING_VALUES_ANALYSIS = "missing_values_analysis"
    OUTLIER_ANALYSIS = "outlier_analysis"
    DISTRIBUTION_ANALYSIS = "distribution_analysis"
    DATA_QUALITY_ASSESSMENT = "data_quality_assessment"
    FEATURE_IMPORTANCE = "feature_importance"
    TREND_ANALYSIS = "trend_analysis"
    BUSINESS_INSIGHTS = "business_insights"

class PromptTemplate(Enum): # This enum seems to be missing in the original code, but used. Added here.
    """Available prompt template types"""
    DATASET_OVERVIEW = "dataset_overview"
    STATISTICAL_SUMMARY = "statistical_summary"
    CORRELATION_INSIGHTS = "correlation_insights"
    MISSING_VALUES_ANALYSIS = "missing_values_analysis"
    OUTLIER_ANALYSIS = "outlier_analysis"
    DISTRIBUTION_ANALYSIS = "distribution_analysis"
    DATA_QUALITY_ASSESSMENT = "data_quality_assessment"
    FEATURE_IMPORTANCE = "feature_importance"
    TREND_ANALYSIS = "trend_analysis"
    BUSINESS_INSIGHTS = "business_insights"

class PromptTemplates: # Placeholder for the PromptTemplates class
    def __init__(self):
        self.templates = {
            PromptTemplate.DATASET_OVERVIEW.value: """
You are an AI assistant providing a comprehensive overview of a dataset.

Dataset Name: {dataset_name}
Rows: {rows}
Columns: {columns}
Sample Column Names: {column_names}
Data Types: {data_types}
Sample Data (First 3 rows):
{sample_data}

Provide a concise, executive summary of the dataset, highlighting its key characteristics.
Focus on:
1. Overall structure (rows, columns).
2. Types of data present (numeric, categorical, etc.).
3. Initial observations from sample data.
4. Any immediate data quality flags (e.g., potential mixed types, many objects).

Respond in exactly this JSON format:
{{
    "overview": {{
        "overview_line_1": "Concise summary line 1",
        "overview_line_2": "Concise summary line 2",
        "key_characteristics": [
            "Characteristic 1",
            "Characteristic 2",
            "Characteristic 3"
        ]
    }}
}}
""",
            PromptTemplate.STATISTICAL_SUMMARY.value: """
You are an AI assistant analyzing the statistical summary of a dataset.

Provide a comprehensive summary of the dataset, including:
- Basic Info: {basic_info}
- Numeric Summary: {numeric_summary}
- Categorical Summary: {categorical_summary}
- Missing Values: {missing_values}
- Correlations: {correlations}

Provide insights that are:
1. Statistically accurate
2. Actionable for decision-making
3. Highlight data quality issues
4. Suggest next steps

Respond in exactly this JSON format:
{{
    "summary": {{
        "summary_points": [
            "Statistical insight 1 with specific numbers and implications",
            "Statistical insight 2 with specific numbers and implications",
            "Statistical insight 3 with specific numbers and implications",
            "Statistical insight 4 with specific numbers and implications",
            "Statistical insight 5 with specific numbers and implications",
            "Statistical insight 6 with specific numbers and implications",
            "Statistical insight 7 with specific numbers and implications",
            "Statistical insight 8 with specific numbers and implications",
            "Statistical insight 9 with specific numbers and implications",
            "Statistical insight 10 with specific numbers and implications"
        ]
    }}
}}
""",
            
                        PromptTemplate.CORRELATION_INSIGHTS.value: """
You are analyzing correlations in a dataset. Provide insights about relationships between variables.

Correlation Data:
{correlation_data}

Generate 5-7 key insights about:
- Strong positive/negative correlations
- Unexpected relationships
- Business implications
- Recommendations for further analysis

Respond in JSON format:
{{
    "insights": [
        "Correlation insight 1",
        "Correlation insight 2",
        "Correlation insight 3",
        "Correlation insight 4",
        "Correlation insight 5"
    ]
}}
""",PromptTemplate.MISSING_VALUES_ANALYSIS.value: """
Analyze missing value patterns and provide recommendations.

Missing Values Data:
{missing_data}

Provide insights about:
- Missing value patterns
- Potential causes
- Impact on analysis
- Recommended handling strategies

Respond in JSON format:
{{
    "analysis": {{
        "patterns": "Description of missing value patterns",
        "recommendations": [
            "Recommendation 1",
            "Recommendation 2",
            "..."
        ]
    }}
}}
""",
            
            PromptTemplate.OUTLIER_ANALYSIS.value: """
Analyze outliers in the dataset and their implications.

Outlier Data:
{outlier_data}

Provide insights about:
- Outlier patterns
- Potential causes
- Business implications
- Recommended actions

Respond in JSON format:
{{
    "outlier_insights": [
        "Outlier insight 1",
        "Outlier insight 2",
        "..."
    ]
}}
""",
            
            PromptTemplate.DISTRIBUTION_ANALYSIS.value: """
Analyze data distributions and their characteristics.

Distribution Data:
{distribution_data}

Provide insights about:
- Distribution shapes
- Normality assessments
- Skewness implications
- Transformation recommendations

Respond in JSON format:
{{
    "distribution_insights": [
        "Distribution insight 1",
        "Distribution insight 2",
        "..."
    ]
}}
""",
            
            PromptTemplate.DATA_QUALITY_ASSESSMENT.value: """
Assess overall data quality and provide recommendations.

Data Quality Metrics:
{quality_metrics}

Evaluate:
- Data completeness
- Consistency issues
- Accuracy concerns
- Reliability assessment

Respond in JSON format:
{{
    "quality_assessment": {{
        "overall_score": "Score out of 10",
        "strengths": ["Strength 1", "Strength 2"],
        "weaknesses": ["Weakness 1", "Weakness 2"],
        "recommendations": ["Recommendation 1", "Recommendation 2"]
    }}
}}
""",
            
            PromptTemplate.FEATURE_IMPORTANCE.value: """
Analyze feature importance and relevance for modeling.

Feature Data:
{feature_data}

Assess:
- Most informative features
- Redundant features
- Feature engineering opportunities
- Modeling recommendations

Respond in JSON format:
{{
    "feature_analysis": {{
        "important_features": ["Feature 1", "Feature 2"],
        "redundant_features": ["Feature 1", "Feature 2"],
        "engineering_suggestions": ["Suggestion 1", "Suggestion 2"]
    }}
}}
""",
            
            PromptTemplate.TREND_ANALYSIS.value: """
Identify trends and patterns in time-series or sequential data.

Trend Data:
{trend_data}

Analyze:
- Temporal patterns
- Seasonal effects
- Growth/decline trends
- Anomalies

Respond in JSON format:
{{
    "trend_insights": [
        "Trend insight 1",
        "Trend insight 2",
        "..."
    ]
}}
""",
            
            PromptTemplate.BUSINESS_INSIGHTS.value: """
Generate business-focused insights and recommendations.

Business Context:
{business_context}

Provide:
- Strategic insights
- Revenue opportunities
- Risk assessments
- Action items

Respond in JSON format:
{{
    "business_insights": {{
        "opportunities": ["Opportunity 1", "Opportunity 2"],
        "risks": ["Risk 1", "Risk 2"],
        "recommendations": ["Action 1", "Action 2"]
    }}
}}
"""
        }

    def get_template(self, template_name: str) -> Optional[str]:
        return self.templates.get(template_name)
    
    def get(self, template_name: str) -> Optional[str]: # Added for compatibility with the original code's usage
        return self.templates.get(template_name)


class AIAgent:
    """Centralized AI agent for generating insights across all EDA steps"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = None
        self._initialize_client()
        self.prompt_templates = PromptTemplates()
    
    def _initialize_client(self):
        """Initialize Gemini AI client"""
        try:
            if not settings.GEMINI_API_KEY:
                self.logger.error("Gemini API key not found in environment variables")
                raise ValueError("GEMINI_API_KEY is required in environment variables")
            
            # Configure the Gemini client with the API key from settings
            genai.configure(api_key=settings.GEMINI_API_KEY)
            
            # Try to use the latest model, fallback to gemini-pro if not available
            try:
                # List available models and find the best one
                models = genai.list_models()
                model_name = 'models/gemini-2.0-flash-exp'  # Default to latest model with models/ prefix
                
                # Check if the model is available
                available_models = [m.name for m in models if 'models/' in m.name]
                self.logger.info(f"Available models: {available_models}")
                
                if any('gemini-2.0-flash' in m for m in available_models):
                    model_name = 'models/gemini-2.0-flash-exp'
                elif any('gemini-1.5-flash' in m for m in available_models):
                    model_name = 'models/gemini-1.5-flash-latest'
                elif any('gemini-1.5-pro' in m for m in available_models):
                    model_name = 'models/gemini-1.5-pro-latest'
                elif any('gemini-pro' in m for m in available_models):
                    model_name = 'models/gemini-pro'
                else:
                    # Use the first available model if none of the preferred ones are found
                    model_name = available_models[0] if available_models else 'models/gemini-pro'
                
                self.client = genai.GenerativeModel(model_name)
                self.logger.info(f"Initialized Gemini AI client with model: {model_name}")
                
            except Exception as model_error:
                self.logger.warning(f"Failed to initialize with latest model, falling back to models/gemini-pro: {str(model_error)}")
                self.client = genai.GenerativeModel('models/gemini-pro')
                self.logger.info("Initialized Gemini AI client with fallback model: models/gemini-pro")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini client: {str(e)}")
            raise
    
    async def analyze(self, analysis_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main analysis method that routes to specific analysis functions
        
        Args:
            analysis_type: Type of analysis to perform (from AnalysisType enum)
            context: Context data needed for the analysis
            
        Returns:
            Dict containing analysis results
        """
        try:
            if not self.client:
                raise RuntimeError("AI client not initialized. Please check your API key.")
            
            # Get the appropriate prompt template
            prompt = self.prompt_templates.get_template(analysis_type)
            if not prompt:
                raise ValueError(f"No prompt template found for analysis type: {analysis_type}")
            
            # Fill template with context data
            formatted_prompt = prompt.format(**context)
            
            # Generate the response using Gemini
            response = self.client.generate_content(formatted_prompt)
            
            # Process and return the response
            return self._process_response(response, analysis_type)
            
        except Exception as e:
            self.logger.error(f"Error in analysis '{analysis_type}': {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "analysis_type": analysis_type
            }
    
    def _process_response(self, response: Any, analysis_type: str) -> Dict[str, Any]:
        """Process the raw response from the AI model"""
        try:
            # Extract text from response (adjust based on actual response structure)
            text_response = response.text if hasattr(response, 'text') else str(response)
            
            # Try to parse as JSON if possible
            try:
                result = json.loads(text_response)
                if not isinstance(result, dict):
                    result = {"insights": result}
            except (json.JSONDecodeError, TypeError):
                result = {"insights": text_response}
            
            # Add metadata
            result.update({
                "success": True,
                "analysis_type": analysis_type,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing response: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to process response: {str(e)}",
                "analysis_type": analysis_type
            }
    
    # Specific analysis methods that can be called directly
    async def analyze_dataset_overview(self, df: pd.DataFrame, dataset_name: str = None) -> Dict[str, Any]:
        """Generate a dataset overview analysis"""
        context = {
            "dataset_name": dataset_name or "Unnamed Dataset",
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": ", ".join(df.columns.tolist()[:10]),  # First 10 columns
            "data_types": str(df.dtypes.to_dict()),
            "sample_data": df.head(3).to_string()
        }
        return await self.analyze(AnalysisType.DATASET_OVERVIEW.value, context)
    
    async def analyze_statistical_summary(self, df: pd.DataFrame, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistical summary analysis"""
        # Import json_utils for proper serialization
        from app.utils.json_utils import serialize_for_json
        
        context = {
            "basic_info": json.dumps(serialize_for_json(stats.get("basic_info", {}))),
            "numeric_summary": json.dumps(serialize_for_json(stats.get("numeric_summary", {}))),
            "categorical_summary": json.dumps(serialize_for_json(stats.get("categorical_summary", {}))),
            "missing_values": json.dumps({k: int(v) for k, v in df.isnull().sum().to_dict().items()}),
            "correlations": json.dumps(serialize_for_json(df.select_dtypes(include=[np.number]).corr().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 1 else {}))
        }
        return await self.analyze(AnalysisType.STATISTICAL_SUMMARY.value, context)
    
    async def analyze_correlations(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Generate correlation analysis"""
        # Get top 10 correlation pairs
        corr_pairs = correlation_matrix.unstack().sort_values(ascending=False)
        corr_pairs = corr_pairs[corr_pairs < 1.0].head(10)  # Exclude self-correlation
        
        context = {
            "correlation_data": corr_pairs.to_string(),
            "top_correlations": corr_pairs.to_dict()
        }
        return await self.analyze(AnalysisType.CORRELATION_INSIGHTS.value, context)
    
    async def analyze_missing_values(self, missing_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate missing values analysis"""
        context = {
            "missing_data": json.dumps(missing_data),
            "total_missing": sum(missing_data.values()),
            "columns_with_missing": [col for col, count in missing_data.items() if count > 0]
        }
        return await self.analyze(AnalysisType.MISSING_VALUES_ANALYSIS.value, context)
    
    async def analyze_outliers(self, df: pd.DataFrame, outliers: Dict[str, Any]) -> Dict[str, Any]:
        """Generate outlier analysis"""
        context = {
            "outlier_data": json.dumps(outliers, indent=2),
            "columns_with_outliers": list(outliers.keys()),
            "total_outliers": sum(len(v) for v in outliers.values())
        }
        return await self.analyze(AnalysisType.OUTLIER_ANALYSIS.value, context)

    async def analyze_distributions(self, distributions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate distribution analysis"""
        context = {
            "distribution_data": json.dumps(distributions, indent=2),
            "columns_analyzed": list(distributions.keys())
        }
        return await self.analyze(AnalysisType.DISTRIBUTION_ANALYSIS.value, context)
    
    async def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data quality assessment"""
        context = {
            "quality_metrics": json.dumps({
                "shape": f"{len(df)} rows × {len(df.columns)} columns",
                "missing_values": df.isnull().sum().to_dict(),
                "duplicate_rows": df.duplicated().sum(),
                "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
            }, indent=2)
        }
        return await self.analyze(AnalysisType.DATA_QUALITY_ASSESSMENT.value, context)
    
    async def analyze_feature_importance(self, feature_importance: Dict[str, float]) -> Dict[str, Any]:
        """Generate feature importance analysis"""
        context = {
            "feature_data": json.dumps(feature_importance, indent=2),
            "top_features": sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        }
        return await self.analyze(AnalysisType.FEATURE_IMPORTANCE.value, context)
    
    async def analyze_trends(self, time_series_data: pd.DataFrame, date_column: str) -> Dict[str, Any]:
        """Generate trend analysis for time series data"""
        context = {
            "trend_data": json.dumps({
                "time_period": {
                    "start": time_series_data[date_column].min().isoformat() if hasattr(time_series_data[date_column].min(), 'isoformat') else str(time_series_data[date_column].min()),
                    "end": time_series_data[date_column].max().isoformat() if hasattr(time_series_data[date_column].max(), 'isoformat') else str(time_series_data[date_column].max())
                },
                "numeric_columns": [col for col in time_series_data.columns 
                                  if col != date_column and pd.api.types.is_numeric_dtype(time_series_data[col])]
            }, indent=2)
        }
        return await self.analyze(AnalysisType.TREND_ANALYSIS.value, context)
    
    async def generate_business_insights(self, df: pd.DataFrame, business_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate business insights from the dataset"""
        context = {
            "business_context": json.dumps({
                "dataset_summary": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "numeric_columns": [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])],
                    "categorical_columns": [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
                },
                "provided_context": business_context or {}
            }, indent=2)
        }
        return await self.analyze(AnalysisType.BUSINESS_INSIGHTS.value, context)

    async def generate_insight(
        self, 
        template_type: PromptTemplate, 
        data_context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate AI insights using specified prompt template
        
        Args:
            template_type: Type of prompt template to use
            data_context: Context data to fill template placeholders
            **kwargs: Additional parameters
            
        Returns:
            Dict containing AI-generated insights
        """
        try:
            if not self.client:
                return self._get_fallback_response(template_type)
            
            # Get prompt template
            template = self.prompt_templates.get(template_type.value)
            if not template:
                raise ValueError(f"Template {template_type.value} not found")
            
            # Fill template with data context
            prompt = template.format(**data_context)
            
            # Generate response
            response = self.client.generate_content(prompt)
            
            if not response or not response.text:
                return self._get_fallback_response(template_type)
            
            # Parse JSON response
            try:
                result = json.loads(response.text.strip())
                return {
                    "success": True,
                    "data": result,
                    "template_used": template_type.value,
                    "timestamp": datetime.utcnow().isoformat()
                }
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw text
                return {
                    "success": True,
                    "data": {"raw_response": response.text.strip()},
                    "template_used": template_type.value,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error generating insight with template {template_type.value}: {str(e)}")
            return self._get_fallback_response(template_type)
    
    def _get_fallback_response(self, template_type: PromptTemplate) -> Dict[str, Any]:
        """Generate fallback response when AI is unavailable"""
        fallbacks = {
            PromptTemplate.DATASET_OVERVIEW.value: {
                "overview": {
                    "overview_line_1": "This dataset contains structured data for comprehensive analysis.",
                    "overview_line_2": "The data can provide valuable insights for decision-making and pattern discovery."
                }
            },
            PromptTemplate.STATISTICAL_SUMMARY.value: {
                "summary": {
                    "summary_points": [
                        "Dataset contains multiple columns with mixed data types requiring analysis",
                        "Statistical distributions vary across different numerical variables",
                        "Data quality assessment shows areas for potential improvement",
                        "Missing value patterns may indicate systematic data collection issues",
                        "Correlation analysis could reveal important relationships between variables",
                        "Outlier detection is recommended to identify anomalous data points",
                        "Feature engineering opportunities exist for enhanced model performance",
                        "Categorical variables show different levels of cardinality",
                        "Numerical ranges vary significantly across different measurements",
                        "Further domain expertise recommended for deeper business insights"
                    ]
                }
            },
            PromptTemplate.CORRELATION_INSIGHTS.value: {
                "insights": [
                    "Correlation analysis requires numerical data processing",
                    "Strong correlations may indicate redundant features",
                    "Weak correlations suggest independent variables"
                ]
            },
            PromptTemplate.MISSING_VALUES_ANALYSIS.value: {
                "analysis": {
                    "patterns": "General patterns of missing values indicate potential data collection gaps.",
                    "recommendations": [
                        "Review data collection processes for completeness.",
                        "Consider imputation strategies for critical missing data.",
                        "Assess the impact of missingness on downstream analysis."
                    ]
                }
            },
            PromptTemplate.OUTLIER_ANALYSIS.value: {
                "outlier_insights": [
                    "Potential outliers detected in numerical features.",
                    "Investigate extreme values for data entry errors or genuine anomalies.",
                    "Determine appropriate handling strategies (e.g., capping, removal) based on domain knowledge."
                ]
            },
            PromptTemplate.DISTRIBUTION_ANALYSIS.value: {
                "distribution_insights": [
                    "Many numerical features exhibit non-normal distributions.",
                    "Skewness in some features may require transformations for certain models.",
                    "Assess the implications of distribution shapes on statistical tests."
                ]
            },
            PromptTemplate.DATA_QUALITY_ASSESSMENT.value: {
                "quality_assessment": {
                    "overall_score": "7/10",
                    "strengths": ["Dataset structure is consistent", "Key identifiers are present"],
                    "weaknesses": ["Some missing values observed", "Potential for duplicates"],
                    "recommendations": ["Implement data validation rules.", "Cleanse missing and duplicate records."]
                }
            },
            PromptTemplate.FEATURE_IMPORTANCE.value: {
                "feature_analysis": {
                    "important_features": ["Feature X", "Feature Y"],
                    "redundant_features": ["Feature A", "Feature B"],
                    "engineering_suggestions": ["Create interaction terms for highly correlated features.", "Derive new features from existing date/time columns."]
                }
            },
            PromptTemplate.TREND_ANALYSIS.value: {
                "trend_insights": [
                    "General upward trend observed over the analysis period.",
                    "No significant seasonality detected, but weekly patterns might exist.",
                    "Consider external factors for sudden spikes or drops."
                ]
            },
            PromptTemplate.BUSINESS_INSIGHTS.value: {
                "business_insights": {
                    "opportunities": ["Optimize product pricing based on demand trends.", "Target specific customer segments with tailored promotions."],
                    "risks": ["Potential for customer churn due to service issues.", "Supply chain disruptions impacting sales."],
                    "recommendations": ["Invest in customer feedback mechanisms.", "Diversify supplier base."]
                }
            }
        }
        
        return {
            "success": False,
            "data": fallbacks.get(template_type.value, {"message": "AI analysis temporarily unavailable"}),
            "template_used": template_type.value,
            "timestamp": datetime.utcnow().isoformat(),
            "fallback": True
        }
    
# Global AI agent instance
ai_agent = AIAgent()