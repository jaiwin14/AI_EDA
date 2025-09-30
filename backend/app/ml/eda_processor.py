"""
EDA Processor for automated exploratory data analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EDAProcessor:
    """Main class for performing exploratory data analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def generate_basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate basic statistical summary of the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dict containing basic statistics
        """
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Basic dataset info
            basic_info = {
                "shape": df.shape,
                "columns": int(len(df.columns)),
                "rows": int(len(df)),
                "numeric_columns": int(len(numeric_cols)),
                "categorical_columns": int(len(categorical_cols)),
                "memory_usage_mb": float(df.memory_usage(deep=True).sum() / (1024 * 1024)),
                "duplicate_rows": int(df.duplicated().sum())
            }
            
            # Numeric summary - convert numpy types to native Python types
            numeric_summary = {}
            if numeric_cols:
                desc = df[numeric_cols].describe()
                numeric_summary = {}
                for col in desc.columns:
                    numeric_summary[col] = {}
                    for stat in desc.index:
                        value = desc.loc[stat, col]
                        if pd.isna(value):
                            numeric_summary[col][stat] = None
                        else:
                            numeric_summary[col][stat] = float(value)
            
            # Categorical summary
            categorical_summary = {}
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                categorical_summary[col] = {
                    "unique_values": int(df[col].nunique()),
                    "most_frequent": value_counts.index[0] if len(value_counts) > 0 else None,
                    "frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    "missing_values": int(df[col].isnull().sum())
                }
            
            return {
                "basic_info": basic_info,
                "numeric_summary": numeric_summary,
                "categorical_summary": categorical_summary
            }
            
        except Exception as e:
            self.logger.error(f"Error in generate_basic_statistics: {str(e)}")
            raise
    
    async def analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze missing values in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dict containing missing value analysis
        """
        try:
            missing_counts = df.isnull().sum()
            missing_percentages = (missing_counts / len(df)) * 100
            
            # Columns with missing values
            columns_with_missing = {}
            for col in df.columns:
                if missing_counts[col] > 0:
                    columns_with_missing[col] = {
                        "count": int(missing_counts[col]),
                        "percentage": float(missing_percentages[col])
                    }
            
            # Missing value patterns
            total_missing = missing_counts.sum()
            total_cells = len(df) * len(df.columns)
            
            return {
                "total_missing": int(total_missing),
                "total_cells": int(total_cells),
                "missing_percentage": float((total_missing / total_cells) * 100),
                "columns_with_missing": columns_with_missing,
                "missing_counts": missing_counts.to_dict(),
                "missing_percentages": missing_percentages.to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"Error in analyze_missing_values: {str(e)}")
            raise
    
    async def analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze correlations between numeric variables with enhanced detection
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dict containing correlation analysis
        """
        try:
            # Convert string numeric columns to actual numeric types
            df_processed = df.copy()
            
            # Try to convert columns that look numeric but are stored as strings
            for col in df_processed.columns:
                if df_processed[col].dtype == 'object':
                    try:
                        # Remove common non-numeric characters and try conversion
                        cleaned_series = df_processed[col].astype(str).str.replace(',', '').str.replace('$', '').str.replace('%', '')
                        numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                        
                        # If more than 50% of values are numeric, treat as numeric
                        if numeric_series.notna().sum() / len(numeric_series) > 0.5:
                            df_processed[col] = numeric_series
                    except:
                        continue
            
            # Get numeric columns including converted ones
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove columns that are mostly NaN or constant
            valid_numeric_cols = []
            for col in numeric_cols:
                col_data = df_processed[col].dropna()
                if len(col_data) > 1 and col_data.nunique() > 1:  # More than 1 unique value
                    valid_numeric_cols.append(col)
            
            if len(valid_numeric_cols) < 2:
                return {
                    "correlation_matrix": {},
                    "strong_correlations": [],
                    "numeric_columns": valid_numeric_cols,
                    "message": f"Not enough valid numeric columns for correlation analysis. Found {len(valid_numeric_cols)} valid numeric columns: {valid_numeric_cols}"
                }
            
            # Calculate correlation matrix using only valid numeric columns
            corr_matrix = df_processed[valid_numeric_cols].corr()
            
            # Find strong correlations (> 0.7 or < -0.7)
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if not pd.isna(corr_val) and abs(corr_val) > 0.7:
                        strong_correlations.append({
                            "variable1": corr_matrix.columns[i],
                            "variable2": corr_matrix.columns[j],
                            "correlation": float(corr_val)
                        })
            
            return {
                "correlation_matrix": corr_matrix.to_dict(),
                "strong_correlations": strong_correlations,
                "numeric_columns": valid_numeric_cols,
                "total_correlations": len(strong_correlations),
                "correlation_summary": {
                    "total_numeric_columns": len(valid_numeric_cols),
                    "strong_positive": len([c for c in strong_correlations if c["correlation"] > 0.7]),
                    "strong_negative": len([c for c in strong_correlations if c["correlation"] < -0.7]),
                    "max_correlation": max([abs(c["correlation"]) for c in strong_correlations]) if strong_correlations else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in analyze_correlations: {str(e)}")
            raise
    
    async def detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect outliers in numeric columns using IQR method
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dict containing outlier analysis
        """
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            outlier_summary = {}
            
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) == 0:
                    continue
                
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                
                outlier_summary[col] = {
                    "outlier_count": len(outliers),
                    "outlier_percentage": (len(outliers) / len(col_data)) * 100,
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "outlier_values": outliers.tolist()[:10]  # First 10 outliers
                }
            
            return {
                "outlier_summary": outlier_summary,
                "total_outliers": sum([info["outlier_count"] for info in outlier_summary.values()])
            }
            
        except Exception as e:
            self.logger.error(f"Error in detect_outliers: {str(e)}")
            raise
    
    async def analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze distributions of numeric variables
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dict containing distribution analysis
        """
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            distribution_summary = {}
            
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) == 0:
                    continue
                
                # Basic statistics
                mean_val = float(col_data.mean())
                median_val = float(col_data.median())
                std_val = float(col_data.std())
                skewness = float(col_data.skew())
                kurtosis = float(col_data.kurtosis())
                
                # Normality test (Shapiro-Wilk for small samples, Anderson-Darling for larger)
                normality_test = "not_tested"
                p_value = None
                
                if len(col_data) <= 5000:
                    try:
                        _, p_value = stats.shapiro(col_data.sample(min(5000, len(col_data))))
                        normality_test = "shapiro_wilk"
                    except:
                        pass
                
                distribution_summary[col] = {
                    "mean": mean_val,
                    "median": median_val,
                    "std": std_val,
                    "skewness": skewness,
                    "kurtosis": kurtosis,
                    "normality_test": normality_test,
                    "normality_p_value": float(p_value) if p_value else None,
                    "is_normal": bool(p_value > 0.05) if p_value else None
                }
            
            return {
                "distribution_summary": distribution_summary,
                "numeric_columns": numeric_cols
            }
            
        except Exception as e:
            self.logger.error(f"Error in analyze_distributions: {str(e)}")
            raise
    
    async def univariate_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform univariate analysis on each column
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dict containing univariate analysis results
        """
        try:
            analysis_results = {}
            
            for col in df.columns:
                col_analysis = {
                    "column_name": col,
                    "data_type": str(df[col].dtype),
                    "non_null_count": int(df[col].count()),
                    "null_count": int(df[col].isnull().sum()),
                    "unique_count": int(df[col].nunique()),
                    "unique_ratio": float(df[col].nunique() / len(df))
                }
                
                if df[col].dtype in ['object', 'category']:
                    # Categorical analysis
                    value_counts = df[col].value_counts()
                    col_analysis.update({
                        "most_frequent": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                        "frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                        "top_values": dict(value_counts.head(5))
                    })
                else:
                    # Numeric analysis
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        col_analysis.update({
                            "min": float(col_data.min()),
                            "max": float(col_data.max()),
                            "mean": float(col_data.mean()),
                            "median": float(col_data.median()),
                            "std": float(col_data.std()) if len(col_data) > 1 else 0.0
                        })
                
                analysis_results[col] = col_analysis
            
            return {
                "univariate_results": analysis_results,
                "total_columns": len(df.columns)
            }
            
        except Exception as e:
            self.logger.error(f"Error in univariate_analysis: {str(e)}")
            raise
    
    async def analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze data types and suggest optimizations
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dict containing data type analysis
        """
        try:
            type_summary = {}
            memory_usage = df.memory_usage(deep=True)
            
            for col in df.columns:
                current_type = str(df[col].dtype)
                current_memory = memory_usage[col]
                
                # Suggest optimizations
                suggestions = []
                if df[col].dtype == 'object':
                    # Check if it can be converted to category
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio < 0.5:
                        suggestions.append("Convert to category to save memory")
                
                elif df[col].dtype in ['int64', 'float64']:
                    # Check if smaller int/float types can be used
                    if df[col].dtype == 'int64':
                        min_val, max_val = df[col].min(), df[col].max()
                        if min_val >= -128 and max_val <= 127:
                            suggestions.append("Can use int8")
                        elif min_val >= -32768 and max_val <= 32767:
                            suggestions.append("Can use int16")
                        elif min_val >= -2147483648 and max_val <= 2147483647:
                            suggestions.append("Can use int32")
                
                type_summary[col] = {
                    "current_type": current_type,
                    "memory_usage_bytes": int(current_memory),
                    "memory_usage_mb": float(current_memory / (1024 * 1024)),
                    "suggestions": suggestions
                }
            
            return {
                "type_summary": type_summary,
                "total_memory_mb": float(memory_usage.sum() / (1024 * 1024))
            }
            
        except Exception as e:
            self.logger.error(f"Error in analyze_data_types: {str(e)}")
            raise
