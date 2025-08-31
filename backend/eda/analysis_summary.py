"""Analysis Summary Module with AI-Generated Insights"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.ai_utils import generate_insight
from utils.serialization import to_json_serializable

def run(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive analysis summary using AI
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing comprehensive analysis summary
    """
    try:
        results = {
            "text": f"Comprehensive analysis summary generated for dataset with {df.shape[0]} rows and {df.shape[1]} columns",
            "dataset_overview": {},
            "data_quality_assessment": {},
            "key_insights": "",
            "recommendations": "",
            "next_steps": ""
        }
        
        # Generate dataset overview
        results["dataset_overview"] = generate_dataset_overview(df)
        
        # Generate data quality assessment
        results["data_quality_assessment"] = assess_data_quality(df)
        
        # Generate AI insights
        insight_prompt = f"""
        Provide a comprehensive analysis summary for this dataset:
        
        Dataset Overview:
        - Shape: {df.shape}
        - Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB
        - Column types: {df.dtypes.value_counts().to_dict()}
        - Missing values: {df.isnull().sum().sum()} total
        
        Data Quality Assessment:
        - Missing data percentage: {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.2f}%
        - Duplicate rows: {df.duplicated().sum()}
        - Unique values per column: {df.nunique().to_dict()}
        
        Please provide:
        1. Key insights about the data structure and characteristics
        2. Data quality issues and their potential impact
        3. Recommendations for data preprocessing and cleaning
        4. Suggestions for further analysis
        5. Potential use cases for this dataset
        6. Next steps for exploratory data analysis
        
        Focus on actionable insights and practical recommendations.
        """
        
        try:
            results["key_insights"] = generate_insight(insight_prompt, context={
                "dataset_overview": results["dataset_overview"],
                "data_quality": results["data_quality_assessment"]
            })
        except Exception as e:
            results["key_insights"] = f"Unable to generate AI insights: {str(e)}"
        
        # Generate recommendations
        results["recommendations"] = generate_recommendations(df, results["data_quality_assessment"])
        
        # Generate next steps
        results["next_steps"] = generate_next_steps(df, results["data_quality_assessment"])
        
        return results
        
    except Exception as e:
        return {
            "text": f"Error in analysis summary: {str(e)}",
            "error": str(e)
        }

def generate_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive dataset overview"""
    
    overview = {
        "basic_info": {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "total_cells": df.shape[0] * df.shape[1]
        },
        "column_types": {
            "numeric": len(df.select_dtypes(include=[np.number]).columns),
            "categorical": len(df.select_dtypes(include=['object', 'category', 'bool']).columns),
            "datetime": len(df.select_dtypes(include=['datetime64']).columns),
            "type_distribution": df.dtypes.value_counts().to_dict()
        },
        "missing_data": {
            "total_missing": df.isnull().sum().sum(),
            "missing_percentage": round((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2),
            "columns_with_missing": int((df.isnull().sum() > 0).sum()),
            "missing_by_column": df.isnull().sum().to_dict()
        },
        "duplicates": {
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_percentage": round((df.duplicated().sum() / len(df)) * 100, 2)
        }
    }
    
    return overview

def assess_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Assess data quality issues"""
    
    quality_assessment = {
        "missing_data_issues": {},
        "duplicate_issues": {},
        "data_type_issues": {},
        "outlier_issues": {},
        "consistency_issues": {},
        "overall_quality_score": 0
    }
    
    # Missing data assessment
    missing_by_column = df.isnull().sum()
    high_missing_columns = missing_by_column[missing_by_column > len(df) * 0.1].to_dict()
    
    quality_assessment["missing_data_issues"] = {
        "high_missing_columns": high_missing_columns,
        "missing_patterns": "Random" if len(high_missing_columns) == 0 else "Systematic",
        "severity": "Low" if len(high_missing_columns) == 0 else "High"
    }
    
    # Duplicate assessment
    duplicate_count = df.duplicated().sum()
    duplicate_percentage = (duplicate_count / len(df)) * 100
    
    quality_assessment["duplicate_issues"] = {
        "duplicate_count": int(duplicate_count),
        "duplicate_percentage": round(duplicate_percentage, 2),
        "severity": "Low" if duplicate_percentage < 5 else "High"
    }
    
    # Data type assessment
    type_issues = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if object column might be numeric
            try:
                pd.to_numeric(df[col], errors='raise')
                type_issues.append(f"{col}: Object column contains numeric data")
            except:
                pass
    
    quality_assessment["data_type_issues"] = {
        "potential_type_conversions": type_issues,
        "severity": "Low" if len(type_issues) == 0 else "Medium"
    }
    
    # Outlier assessment (simplified)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    outlier_columns = []
    
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)][col]
        if len(outliers) > len(df) * 0.05:  # More than 5% outliers
            outlier_columns.append(col)
    
    quality_assessment["outlier_issues"] = {
        "columns_with_outliers": outlier_columns,
        "severity": "Low" if len(outlier_columns) == 0 else "Medium"
    }
    
    # Calculate overall quality score
    score = 100
    
    # Deduct points for missing data
    missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    score -= missing_percentage * 2  # 2 points per percentage
    
    # Deduct points for duplicates
    duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
    score -= duplicate_percentage * 3  # 3 points per percentage
    
    # Deduct points for data type issues
    score -= len(type_issues) * 5  # 5 points per type issue
    
    # Deduct points for outliers
    score -= len(outlier_columns) * 3  # 3 points per outlier column
    
    quality_assessment["overall_quality_score"] = max(0, round(score, 1))
    
    return quality_assessment

def generate_recommendations(df: pd.DataFrame, quality_assessment: Dict) -> str:
    """Generate data preprocessing recommendations"""
    
    recommendations = []
    
    # Missing data recommendations
    missing_issues = quality_assessment["missing_data_issues"]
    if missing_issues["severity"] == "High":
        recommendations.append("• Address missing data in high-missing columns through imputation or removal")
    elif missing_issues["severity"] == "Low":
        recommendations.append("• Missing data levels are acceptable; consider simple imputation methods")
    
    # Duplicate recommendations
    duplicate_issues = quality_assessment["duplicate_issues"]
    if duplicate_issues["severity"] == "High":
        recommendations.append("• Remove duplicate rows to improve data quality")
    elif duplicate_issues["severity"] == "Low":
        recommendations.append("• Duplicate levels are acceptable; verify if duplicates are intentional")
    
    # Data type recommendations
    type_issues = quality_assessment["data_type_issues"]
    if type_issues["severity"] != "Low":
        recommendations.append("• Convert object columns with numeric data to appropriate numeric types")
    
    # Outlier recommendations
    outlier_issues = quality_assessment["outlier_issues"]
    if outlier_issues["severity"] != "Low":
        recommendations.append("• Investigate and handle outliers in identified columns")
    
    # General recommendations
    recommendations.extend([
        "• Perform exploratory data analysis to understand data distributions",
        "• Check for data consistency and business logic validation",
        "• Consider feature engineering based on domain knowledge",
        "• Evaluate the need for data standardization or normalization"
    ])
    
    return "\n".join(recommendations)

def generate_next_steps(df: pd.DataFrame, quality_assessment: Dict) -> str:
    """Generate next steps for analysis"""
    
    next_steps = [
        "1. Data Cleaning:",
        "   • Handle missing values using appropriate strategies",
        "   • Remove or investigate duplicate records",
        "   • Convert data types where necessary",
        "",
        "2. Exploratory Data Analysis:",
        "   • Analyze distributions of numeric variables",
        "   • Examine categorical variable frequencies",
        "   • Create correlation analysis for numeric variables",
        "   • Generate summary statistics",
        "",
        "3. Data Visualization:",
        "   • Create histograms and box plots for numeric variables",
        "   • Generate bar charts and pie charts for categorical variables",
        "   • Build correlation heatmaps",
        "   • Create scatter plots for key variable relationships",
        "",
        "4. Feature Engineering:",
        "   • Create new features based on domain knowledge",
        "   • Handle outliers appropriately",
        "   • Encode categorical variables",
        "   • Scale numeric variables if needed",
        "",
        "5. Advanced Analysis:",
        "   • Perform dimensionality reduction if needed",
        "   • Conduct statistical tests",
        "   • Build predictive models",
        "   • Validate results and assumptions"
    ]
    
    return "\n".join(next_steps)
