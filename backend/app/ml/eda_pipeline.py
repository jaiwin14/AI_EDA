"""
Automated EDA Pipeline
Comprehensive exploratory data analysis with statistical tests and visualizations
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import chi2_contingency, normaltest, jarque_bera
import logging
from typing import Dict, Any, List, Tuple, Optional
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class EDAProcessor:
    """Comprehensive EDA processing pipeline"""
    
    def __init__(self):
        self.results = {}
        
    async def run_full_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run complete EDA pipeline"""
        
        logger.info("Starting comprehensive EDA analysis")
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "dataset_shape": df.shape,
            "basic_statistics": await self.generate_basic_statistics(df),
            "missing_values": await self.analyze_missing_values(df),
            "data_types": await self.analyze_data_types(df),
            "univariate_analysis": await self.univariate_analysis(df),
            "bivariate_analysis": await self.bivariate_analysis(df),
            "correlations": await self.analyze_correlations(df),
            "outliers": await self.detect_outliers(df),
            "distribution_analysis": await self.analyze_distributions(df),
            "feature_importance": await self.calculate_feature_importance(df)
        }
        
        logger.info("EDA analysis completed")
        return results
    
    async def generate_basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive basic statistics"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        stats = {
            "overview": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "numeric_columns": len(numeric_cols),
                "categorical_columns": len(categorical_cols),
                "datetime_columns": len(datetime_cols),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                "duplicate_rows": df.duplicated().sum()
            },
            "numeric_summary": {},
            "categorical_summary": {},
            "column_info": []
        }
        
        # Numeric columns statistics
        if numeric_cols:
            numeric_df = df[numeric_cols]
            stats["numeric_summary"] = {
                "describe": numeric_df.describe().to_dict(),
                "skewness": numeric_df.skew().to_dict(),
                "kurtosis": numeric_df.kurtosis().to_dict(),
                "variance": numeric_df.var().to_dict(),
                "std_dev": numeric_df.std().to_dict()
            }
        
        # Categorical columns statistics
        if categorical_cols:
            cat_stats = {}
            for col in categorical_cols:
                cat_stats[col] = {
                    "unique_count": df[col].nunique(),
                    "unique_ratio": df[col].nunique() / len(df),
                    "most_frequent": df[col].mode().iloc[0] if not df[col].mode().empty else None,
                    "most_frequent_count": df[col].value_counts().iloc[0] if not df[col].empty else 0,
                    "value_counts": df[col].value_counts().head(10).to_dict()
                }
            stats["categorical_summary"] = cat_stats
        
        # Column-wise information
        for col in df.columns:
            col_info = {
                "column": col,
                "dtype": str(df[col].dtype),
                "non_null_count": df[col].count(),
                "null_count": df[col].isnull().sum(),
                "null_percentage": (df[col].isnull().sum() / len(df)) * 100,
                "unique_count": df[col].nunique(),
                "unique_ratio": df[col].nunique() / len(df)
            }
            
            if col in numeric_cols:
                col_info.update({
                    "min": float(df[col].min()) if pd.notna(df[col].min()) else None,
                    "max": float(df[col].max()) if pd.notna(df[col].max()) else None,
                    "mean": float(df[col].mean()) if pd.notna(df[col].mean()) else None,
                    "median": float(df[col].median()) if pd.notna(df[col].median()) else None,
                    "std": float(df[col].std()) if pd.notna(df[col].std()) else None,
                    "skewness": float(df[col].skew()) if pd.notna(df[col].skew()) else None,
                    "kurtosis": float(df[col].kurtosis()) if pd.notna(df[col].kurtosis()) else None
                })
            
            stats["column_info"].append(col_info)
        
        return stats
    
    async def analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive missing values analysis"""
        
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        analysis = {
            "total_missing": int(missing_counts.sum()),
            "total_cells": len(df) * len(df.columns),
            "overall_missing_percentage": float((missing_counts.sum() / (len(df) * len(df.columns))) * 100),
            "columns_with_missing": missing_counts[missing_counts > 0].to_dict(),
            "missing_percentages": missing_percentages[missing_percentages > 0].to_dict(),
            "missing_patterns": {},
            "recommendations": []
        }
        
        # Missing value patterns
        if missing_counts.sum() > 0:
            # Create missing value pattern matrix
            missing_matrix = df.isnull()
            patterns = missing_matrix.value_counts().head(10)
            analysis["missing_patterns"] = {
                str(pattern): count for pattern, count in patterns.items()
            }
        
        # Recommendations based on missing data
        high_missing_cols = missing_percentages[missing_percentages > 50].index.tolist()
        medium_missing_cols = missing_percentages[(missing_percentages > 20) & (missing_percentages <= 50)].index.tolist()
        low_missing_cols = missing_percentages[(missing_percentages > 0) & (missing_percentages <= 20)].index.tolist()
        
        if high_missing_cols:
            analysis["recommendations"].append({
                "type": "high_missing",
                "columns": high_missing_cols,
                "suggestion": "Consider dropping these columns or investigating data collection issues"
            })
        
        if medium_missing_cols:
            analysis["recommendations"].append({
                "type": "medium_missing", 
                "columns": medium_missing_cols,
                "suggestion": "Consider advanced imputation techniques or domain-specific handling"
            })
        
        if low_missing_cols:
            analysis["recommendations"].append({
                "type": "low_missing",
                "columns": low_missing_cols,
                "suggestion": "Simple imputation (mean/median/mode) should work well"
            })
        
        return analysis
    
    async def analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze and suggest optimal data types"""
        
        analysis = {
            "current_dtypes": df.dtypes.astype(str).to_dict(),
            "suggested_dtypes": {},
            "memory_optimization": {},
            "type_recommendations": []
        }
        
        for col in df.columns:
            current_dtype = df[col].dtype
            suggested_dtype = current_dtype
            memory_savings = 0
            
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if integer can be downcasted
                if pd.api.types.is_integer_dtype(df[col]):
                    min_val = df[col].min()
                    max_val = df[col].max()
                    
                    if min_val >= 0:  # Unsigned integers
                        if max_val < 256:
                            suggested_dtype = 'uint8'
                        elif max_val < 65536:
                            suggested_dtype = 'uint16'
                        elif max_val < 4294967296:
                            suggested_dtype = 'uint32'
                    else:  # Signed integers
                        if min_val >= -128 and max_val < 128:
                            suggested_dtype = 'int8'
                        elif min_val >= -32768 and max_val < 32768:
                            suggested_dtype = 'int16'
                        elif min_val >= -2147483648 and max_val < 2147483648:
                            suggested_dtype = 'int32'
                
                # Check if float can be downcasted
                elif pd.api.types.is_float_dtype(df[col]):
                    if df[col].dtype == 'float64':
                        # Check if values fit in float32
                        try:
                            if np.allclose(df[col].dropna(), df[col].dropna().astype('float32'), equal_nan=True):
                                suggested_dtype = 'float32'
                        except:
                            pass
            
            elif pd.api.types.is_object_dtype(df[col]):
                # Check if string column can be categorical
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    suggested_dtype = 'category'
            
            analysis["suggested_dtypes"][col] = str(suggested_dtype)
            
            # Calculate potential memory savings
            if str(suggested_dtype) != str(current_dtype):
                current_memory = df[col].memory_usage(deep=True)
                try:
                    temp_series = df[col].astype(suggested_dtype)
                    new_memory = temp_series.memory_usage(deep=True)
                    memory_savings = current_memory - new_memory
                except:
                    memory_savings = 0
                
                analysis["memory_optimization"][col] = {
                    "current_memory": current_memory,
                    "optimized_memory": current_memory - memory_savings,
                    "savings_bytes": memory_savings,
                    "savings_percentage": (memory_savings / current_memory) * 100 if current_memory > 0 else 0
                }
        
        return analysis
    
    async def univariate_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive univariate analysis for each column"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        analysis = {
            "numeric_analysis": {},
            "categorical_analysis": {},
            "distribution_tests": {},
            "outlier_summary": {}
        }
        
        # Numeric columns analysis
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue
                
            col_analysis = {
                "basic_stats": {
                    "count": len(series),
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "mode": float(series.mode().iloc[0]) if not series.mode().empty else None,
                    "std": float(series.std()),
                    "var": float(series.var()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "range": float(series.max() - series.min()),
                    "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
                    "skewness": float(series.skew()),
                    "kurtosis": float(series.kurtosis())
                },
                "percentiles": {
                    f"p{p}": float(series.quantile(p/100)) 
                    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
                },
                "distribution_shape": self._analyze_distribution_shape(series),
                "outliers": self._detect_outliers_iqr(series)
            }
            
            # Normality tests
            if len(series) >= 8:  # Minimum sample size for tests
                try:
                    # Shapiro-Wilk test (for smaller samples)
                    if len(series) <= 5000:
                        shapiro_stat, shapiro_p = stats.shapiro(series)
                        col_analysis["normality_tests"] = {
                            "shapiro_wilk": {
                                "statistic": float(shapiro_stat),
                                "p_value": float(shapiro_p),
                                "is_normal": shapiro_p > 0.05
                            }
                        }
                    
                    # Jarque-Bera test
                    jb_stat, jb_p = jarque_bera(series)
                    if "normality_tests" not in col_analysis:
                        col_analysis["normality_tests"] = {}
                    col_analysis["normality_tests"]["jarque_bera"] = {
                        "statistic": float(jb_stat),
                        "p_value": float(jb_p),
                        "is_normal": jb_p > 0.05
                    }
                except:
                    pass
            
            analysis["numeric_analysis"][col] = col_analysis
        
        # Categorical columns analysis
        for col in categorical_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            
            value_counts = series.value_counts()
            
            col_analysis = {
                "unique_count": len(value_counts),
                "unique_ratio": len(value_counts) / len(series),
                "most_frequent": {
                    "value": value_counts.index[0],
                    "count": int(value_counts.iloc[0]),
                    "percentage": float((value_counts.iloc[0] / len(series)) * 100)
                },
                "least_frequent": {
                    "value": value_counts.index[-1],
                    "count": int(value_counts.iloc[-1]),
                    "percentage": float((value_counts.iloc[-1] / len(series)) * 100)
                },
                "top_categories": value_counts.head(10).to_dict(),
                "category_distribution": {
                    "entropy": float(stats.entropy(value_counts.values)),
                    "concentration": float(value_counts.iloc[0] / value_counts.sum()),
                    "balance_ratio": float(value_counts.iloc[-1] / value_counts.iloc[0])
                }
            }
            
            analysis["categorical_analysis"][col] = col_analysis
        
        return analysis
    
    async def bivariate_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Bivariate analysis between column pairs"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        analysis = {
            "numeric_pairs": {},
            "categorical_pairs": {},
            "mixed_pairs": {},
            "correlation_summary": {}
        }
        
        # Numeric-Numeric pairs
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    pair_key = f"{col1}_vs_{col2}"
                    
                    # Calculate correlations
                    pearson_corr, pearson_p = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
                    spearman_corr, spearman_p = stats.spearmanr(df[col1].dropna(), df[col2].dropna())
                    
                    analysis["numeric_pairs"][pair_key] = {
                        "pearson_correlation": {
                            "coefficient": float(pearson_corr),
                            "p_value": float(pearson_p),
                            "significant": pearson_p < 0.05
                        },
                        "spearman_correlation": {
                            "coefficient": float(spearman_corr),
                            "p_value": float(spearman_p),
                            "significant": spearman_p < 0.05
                        },
                        "relationship_strength": self._interpret_correlation(abs(pearson_corr))
                    }
        
        # Categorical-Categorical pairs (Chi-square test)
        if len(categorical_cols) >= 2:
            for i, col1 in enumerate(categorical_cols):
                for col2 in categorical_cols[i+1:]:
                    pair_key = f"{col1}_vs_{col2}"
                    
                    try:
                        contingency_table = pd.crosstab(df[col1], df[col2])
                        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                        
                        # Cramér's V for effect size
                        n = contingency_table.sum().sum()
                        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                        
                        analysis["categorical_pairs"][pair_key] = {
                            "chi_square": {
                                "statistic": float(chi2),
                                "p_value": float(p_value),
                                "degrees_of_freedom": int(dof),
                                "significant": p_value < 0.05
                            },
                            "cramers_v": float(cramers_v),
                            "association_strength": self._interpret_cramers_v(cramers_v)
                        }
                    except:
                        pass
        
        return analysis
    
    async def analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive correlation analysis"""
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {"message": "No numeric columns for correlation analysis"}
        
        # Calculate different correlation matrices
        pearson_corr = numeric_df.corr(method='pearson')
        spearman_corr = numeric_df.corr(method='spearman')
        kendall_corr = numeric_df.corr(method='kendall')
        
        analysis = {
            "pearson_correlation": pearson_corr.to_dict(),
            "spearman_correlation": spearman_corr.to_dict(),
            "kendall_correlation": kendall_corr.to_dict(),
            "strong_correlations": {},
            "correlation_summary": {}
        }
        
        # Find strong correlations
        for method, corr_matrix in [("pearson", pearson_corr), ("spearman", spearman_corr)]:
            strong_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) >= 0.7:  # Strong correlation threshold
                        strong_pairs.append({
                            "feature_1": corr_matrix.columns[i],
                            "feature_2": corr_matrix.columns[j],
                            "correlation": float(corr_val),
                            "strength": self._interpret_correlation(abs(corr_val))
                        })
            
            analysis["strong_correlations"][method] = strong_pairs
        
        return analysis
    
    async def detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive outlier detection using multiple methods"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        analysis = {
            "outlier_summary": {},
            "outlier_methods": {},
            "recommendations": []
        }
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            
            col_outliers = {
                "iqr_method": self._detect_outliers_iqr(series),
                "z_score_method": self._detect_outliers_zscore(series),
                "modified_z_score": self._detect_outliers_modified_zscore(series)
            }
            
            # Combine outlier detection results
            all_outlier_indices = set()
            for method_results in col_outliers.values():
                all_outlier_indices.update(method_results.get("outlier_indices", []))
            
            outlier_percentage = (len(all_outlier_indices) / len(series)) * 100
            
            analysis["outlier_summary"][col] = {
                "total_outliers": len(all_outlier_indices),
                "outlier_percentage": float(outlier_percentage),
                "outlier_indices": list(all_outlier_indices),
                "methods_agreement": len([m for m in col_outliers.values() if m.get("outlier_indices")]),
                "severity": "high" if outlier_percentage > 10 else "medium" if outlier_percentage > 5 else "low"
            }
            
            analysis["outlier_methods"][col] = col_outliers
            
            # Recommendations
            if outlier_percentage > 15:
                analysis["recommendations"].append({
                    "column": col,
                    "issue": "High outlier percentage",
                    "suggestion": "Consider robust scaling or outlier treatment methods"
                })
        
        return analysis
    
    async def analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze statistical distributions of numeric columns"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        analysis = {
            "distribution_analysis": {},
            "normality_summary": {},
            "transformation_suggestions": []
        }
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 3:
                continue
            
            col_analysis = {
                "distribution_shape": self._analyze_distribution_shape(series),
                "normality_tests": {},
                "suggested_transformations": []
            }
            
            # Normality tests
            if len(series) >= 8:
                try:
                    # Jarque-Bera test
                    jb_stat, jb_p = jarque_bera(series)
                    col_analysis["normality_tests"]["jarque_bera"] = {
                        "statistic": float(jb_stat),
                        "p_value": float(jb_p),
                        "is_normal": jb_p > 0.05
                    }
                    
                    # D'Agostino's normality test
                    da_stat, da_p = normaltest(series)
                    col_analysis["normality_tests"]["dagostino"] = {
                        "statistic": float(da_stat),
                        "p_value": float(da_p),
                        "is_normal": da_p > 0.05
                    }
                except:
                    pass
            
            # Suggest transformations based on skewness
            skewness = series.skew()
            if abs(skewness) > 1:
                if skewness > 1:
                    col_analysis["suggested_transformations"].extend([
                        "log_transform", "sqrt_transform", "box_cox"
                    ])
                elif skewness < -1:
                    col_analysis["suggested_transformations"].extend([
                        "square_transform", "exponential_transform"
                    ])
            
            analysis["distribution_analysis"][col] = col_analysis
        
        return analysis
    
    async def calculate_feature_importance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate feature importance using various methods"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return {"message": "Insufficient numeric columns for feature importance analysis"}
        
        analysis = {
            "variance_analysis": {},
            "correlation_importance": {},
            "mutual_information": {}
        }
        
        # Variance-based importance
        for col in numeric_cols:
            variance = df[col].var()
            analysis["variance_analysis"][col] = {
                "variance": float(variance),
                "coefficient_of_variation": float(df[col].std() / df[col].mean()) if df[col].mean() != 0 else 0
            }
        
        # Correlation-based importance (average absolute correlation with other features)
        corr_matrix = df[numeric_cols].corr().abs()
        for col in numeric_cols:
            avg_corr = corr_matrix[col].drop(col).mean()
            analysis["correlation_importance"][col] = float(avg_corr)
        
        return analysis
    
    def _analyze_distribution_shape(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze the shape characteristics of a distribution"""
        
        skewness = series.skew()
        kurtosis = series.kurtosis()
        
        # Interpret skewness
        if abs(skewness) < 0.5:
            skew_interpretation = "approximately_symmetric"
        elif skewness > 0.5:
            skew_interpretation = "right_skewed"
        else:
            skew_interpretation = "left_skewed"
        
        # Interpret kurtosis
        if abs(kurtosis) < 0.5:
            kurt_interpretation = "mesokurtic"
        elif kurtosis > 0.5:
            kurt_interpretation = "leptokurtic"
        else:
            kurt_interpretation = "platykurtic"
        
        return {
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "skewness_interpretation": skew_interpretation,
            "kurtosis_interpretation": kurt_interpretation
        }
    
    def _detect_outliers_iqr(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        return {
            "method": "IQR",
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_count": len(outliers),
            "outlier_indices": outliers.index.tolist(),
            "outlier_values": outliers.tolist()
        }
    
    def _detect_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> Dict[str, Any]:
        """Detect outliers using Z-score method"""
        
        z_scores = np.abs(stats.zscore(series))
        outliers = series[z_scores > threshold]
        
        return {
            "method": "Z-Score",
            "threshold": threshold,
            "outlier_count": len(outliers),
            "outlier_indices": outliers.index.tolist(),
            "outlier_values": outliers.tolist()
        }
    
    def _detect_outliers_modified_zscore(self, series: pd.Series, threshold: float = 3.5) -> Dict[str, Any]:
        """Detect outliers using Modified Z-score method"""
        
        median = series.median()
        mad = np.median(np.abs(series - median))
        
        if mad == 0:
            return {
                "method": "Modified Z-Score",
                "threshold": threshold,
                "outlier_count": 0,
                "outlier_indices": [],
                "outlier_values": []
            }
        
        modified_z_scores = 0.6745 * (series - median) / mad
        outliers = series[np.abs(modified_z_scores) > threshold]
        
        return {
            "method": "Modified Z-Score",
            "threshold": threshold,
            "outlier_count": len(outliers),
            "outlier_indices": outliers.index.tolist(),
            "outlier_values": outliers.tolist()
        }
    
    def _interpret_correlation(self, corr_value: float) -> str:
        """Interpret correlation strength"""
        
        if corr_value >= 0.9:
            return "very_strong"
        elif corr_value >= 0.7:
            return "strong"
        elif corr_value >= 0.5:
            return "moderate"
        elif corr_value >= 0.3:
            return "weak"
        else:
            return "very_weak"
    
    def _interpret_cramers_v(self, cramers_v: float) -> str:
        """Interpret Cramér's V association strength"""
        
        if cramers_v >= 0.6:
            return "strong"
        elif cramers_v >= 0.3:
            return "moderate"
        elif cramers_v >= 0.1:
            return "weak"
        else:
            return "very_weak"
