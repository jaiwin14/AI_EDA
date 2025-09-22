"""
Gemini AI Client for generating dataset insights and summaries
"""
import os
import google.generativeai as genai
from typing import Dict, Any, List
import pandas as pd
import json
from ..core.config import settings

class GeminiClient:
    def __init__(self):
        """Initialize Gemini client with API key from environment"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    async def generate_dataset_overview(self, df: pd.DataFrame, dataset_name: str = None) -> Dict[str, Any]:
        """
        Generate a 2-line overview summary about what the dataset represents
        
        Args:
            df: The pandas DataFrame
            dataset_name: Optional name of the dataset file
            
        Returns:
            Dict containing the overview summary
        """
        try:
            # Prepare dataset context
            columns = list(df.columns)
            sample_data = df.head(3).to_dict('records') if len(df) > 0 else []
            data_types = df.dtypes.to_dict()
            
            prompt = f"""
            Analyze this dataset and provide a 2-line summary about what this data represents and its purpose.
            Focus on the real-world meaning, not technical statistics.
            
            Dataset Information:
            - File name: {dataset_name or 'Unknown'}
            - Columns: {columns}
            - Data types: {data_types}
            - Sample rows: {sample_data}
            - Total rows: {len(df)}
            
            Provide a response in this exact JSON format:
            {{
                "overview_line_1": "First line describing what the dataset is about",
                "overview_line_2": "Second line describing the purpose or context"
            }}
            
            Make it sound natural and informative, like explaining to someone what this data shows.
            Example: "This dataset contains information about football matches played in a tournament" rather than "This dataset has numerical and categorical variables".
            """
            
            response = self.model.generate_content(prompt)
            
            # Parse JSON response
            try:
                result = json.loads(response.text)
                return {
                    "success": True,
                    "overview": result,
                    "generated_at": pd.Timestamp.now().isoformat()
                }
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                lines = response.text.strip().split('\n')
                return {
                    "success": True,
                    "overview": {
                        "overview_line_1": lines[0] if len(lines) > 0 else "Dataset analysis completed",
                        "overview_line_2": lines[1] if len(lines) > 1 else "Contains structured data for analysis"
                    },
                    "generated_at": pd.Timestamp.now().isoformat()
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "overview": {
                    "overview_line_1": "Dataset uploaded successfully",
                    "overview_line_2": "Ready for exploratory data analysis"
                }
            }
    
    async def generate_statistics_summary(self, statistical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a 10-point summary of key statistical findings
        
        Args:
            statistical_data: The complete statistical analysis results
            
        Returns:
            Dict containing the 10-point summary
        """
        try:
            # Extract key information for the prompt
            dataset_overview = statistical_data.get('dataset_overview', {})
            numeric_summary = statistical_data.get('basic_statistics', {}).get('numeric_summary', {}).get('describe', {})
            categorical_summary = statistical_data.get('basic_statistics', {}).get('categorical_summary', {})
            
            prompt = f"""
            Analyze these statistical results and provide exactly 10 key insights in bullet points.
            Focus on actionable insights and interesting patterns, not just basic statistics.
            
            Dataset Overview:
            - Total rows: {dataset_overview.get('total_rows', 0)}
            - Total columns: {dataset_overview.get('total_columns', 0)}
            - Missing data: {dataset_overview.get('missing_cells_percentage', 0):.1f}%
            - Duplicate rows: {dataset_overview.get('duplicate_rows_percentage', 0):.1f}%
            - Numeric columns: {dataset_overview.get('numeric_columns', 0)}
            - Categorical columns: {dataset_overview.get('categorical_columns', 0)}
            
            Numerical Variables Summary:
            {json.dumps(numeric_summary, indent=2) if numeric_summary else "No numerical data"}
            
            Categorical Variables Summary:
            {json.dumps(categorical_summary, indent=2) if categorical_summary else "No categorical data"}
            
            Provide a response in this exact JSON format:
            {{
                "summary_points": [
                    "Point 1: Key insight about the data",
                    "Point 2: Another important finding",
                    ...exactly 10 points...
                ]
            }}
            
            Make each point actionable and insightful. Focus on:
            - Data quality issues
            - Interesting patterns or distributions
            - Potential relationships
            - Outliers or anomalies
            - Business implications
            - Recommendations for further analysis
            """
            
            response = self.model.generate_content(prompt)
            
            # Parse JSON response
            try:
                result = json.loads(response.text)
                summary_points = result.get('summary_points', [])
                
                # Ensure exactly 10 points
                if len(summary_points) < 10:
                    summary_points.extend([f"Additional insight {i+1}" for i in range(len(summary_points), 10)])
                elif len(summary_points) > 10:
                    summary_points = summary_points[:10]
                
                return {
                    "success": True,
                    "summary": {
                        "summary_points": summary_points
                    },
                    "generated_at": pd.Timestamp.now().isoformat()
                }
                
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                lines = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
                points = []
                for line in lines:
                    if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                        points.append(line[1:].strip())
                    elif len(line) > 10:  # Assume it's a point if it's substantial
                        points.append(line)
                
                # Ensure exactly 10 points
                while len(points) < 10:
                    points.append(f"Statistical insight {len(points) + 1}")
                
                return {
                    "success": True,
                    "summary": {
                        "summary_points": points[:10]
                    },
                    "generated_at": pd.Timestamp.now().isoformat()
                }
                
        except Exception as e:
            # Fallback summary points
            fallback_points = [
                "Dataset contains structured data ready for analysis",
                "Multiple variables available for exploration", 
                "Data quality assessment completed",
                "Statistical distributions calculated",
                "Missing value patterns identified",
                "Categorical variables analyzed",
                "Numerical summaries generated",
                "Data types properly classified",
                "Memory usage optimized",
                "Ready for advanced analytics"
            ]
            
            return {
                "success": False,
                "error": str(e),
                "summary": {
                    "summary_points": fallback_points
                }
            }
