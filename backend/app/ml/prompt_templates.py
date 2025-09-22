"""
Prompt templates for different AI analysis steps
Each template includes placeholders that will be filled with actual data
"""
from typing import Dict, Any

class PromptTemplates:
    """Centralized prompt templates for all AI analysis steps"""
    
    @staticmethod
    def get_template(template_name: str, context: Dict[str, Any]) -> str:
        """
        Get the appropriate prompt template based on the analysis type
        
        Args:
            template_name: Name of the template to use
            context: Dictionary containing data needed for the template
            
        Returns:
            Formatted prompt string
        """
        templates = {
            "dataset_overview": PromptTemplates._dataset_overview,
            "statistical_summary": PromptTemplates._statistical_summary,
            "correlation_insights": PromptTemplates._correlation_insights,
            "missing_values_analysis": PromptTemplates._missing_values_analysis,
            "outlier_analysis": PromptTemplates._outlier_analysis,
            "distribution_analysis": PromptTemplates._distribution_analysis,
            "data_quality_assessment": PromptTemplates._data_quality_assessment,
            "feature_importance": PromptTemplates._feature_importance,
            "trend_analysis": PromptTemplates._trend_analysis,
            "business_insights": PromptTemplates._business_insights
        }
        
        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}")
            
        return templates[template_name](context)
    
    @staticmethod
    def _dataset_overview(context: Dict[str, Any]) -> str:
        """Template for dataset overview analysis"""
        return f"""
        You are a data analysis expert. Analyze this dataset and provide a comprehensive overview.
        
        Dataset Information:
        - Name: {context.get('dataset_name', 'Unnamed Dataset')}
        - Shape: {context.get('shape', 'N/A')} (rows x columns)
        - Columns: {', '.join(context.get('columns', []))}
        
        Please provide:
        1. A brief description of what this dataset represents in real-world terms
        2. The main purpose of this dataset
        3. Any notable patterns or characteristics at first glance
        4. Potential use cases for this data
        
        Format your response in markdown with clear section headers.
        """
    
    @staticmethod
    def _statistical_summary(context: Dict[str, Any]) -> str:
        """Template for statistical summary analysis"""
        return f"""
        Analyze the statistical properties of this dataset and provide key insights.
        
        Dataset: {context.get('dataset_name', 'Unnamed Dataset')}
        
        Statistics Summary:
        {context.get('statistics', 'No statistics provided')}
        
        Please analyze and provide:
        1. Key statistical measures and their implications
        2. Notable patterns in the data distribution
        3. Any potential data quality issues
        4. Recommendations for further analysis
        
        Format your response in markdown with clear section headers.
        """
    
    @staticmethod
    def _correlation_insights(context: Dict[str, Any]) -> str:
        """Template for correlation analysis"""
        return f"""
        Analyze the correlations in this dataset and provide business insights.
        
        Dataset: {context.get('dataset_name', 'Unnamed Dataset')}
        
        Correlation Matrix (top 10 pairs):
        {context.get('correlation_matrix', 'No correlation data provided')}
        
        Please analyze and provide:
        1. The strongest positive and negative correlations
        2. Potential business implications of these relationships
        3. Any surprising or unexpected correlations
        4. Recommendations for further investigation
        
        Format your response in markdown with clear section headers.
        """
    
    # Add more template methods for other analysis types...
    
    @staticmethod
    def _missing_values_analysis(context: Dict[str, Any]) -> str:
        return """
        Analyze the missing values in the dataset and provide recommendations.
        
        [Your prompt here...]
        """
        
    @staticmethod
    def _outlier_analysis(context: Dict[str, Any]) -> str:
        return """
        Analyze the outliers in the dataset and provide insights.
        
        [Your prompt here...]
        """
        
    @staticmethod
    def _distribution_analysis(context: Dict[str, Any]) -> str:
        return """
        Analyze the distributions of variables in the dataset.
        
        [Your prompt here...]
        """
        
    @staticmethod
    def _data_quality_assessment(context: Dict[str, Any]) -> str:
        return """
        Assess the overall data quality of the dataset.
        
        [Your prompt here...]
        """
        
    @staticmethod
    def _feature_importance(context: Dict[str, Any]) -> str:
        return """
        Analyze and explain the importance of different features.
        
        [Your prompt here...]
        """
        
    @staticmethod
    def _trend_analysis(context: Dict[str, Any]) -> str:
        return """
        Analyze trends in the dataset over time or other dimensions.
        
        [Your prompt here...]
        """
        
    @staticmethod
    def _business_insights(context: Dict[str, Any]) -> str:
        return """
        Provide actionable business insights from the dataset.
        
        [Your prompt here...]
        """
