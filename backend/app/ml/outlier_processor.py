"""
Advanced Outlier Detection and Treatment Processor
Provides comprehensive outlier analysis, detection methods, and multiple treatment options
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from enum import Enum
import logging
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class OutlierMethod(Enum):
    """Available outlier detection methods"""
    Z_SCORE = "z_score"
    MODIFIED_Z_SCORE = "modified_z_score"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    PERCENTILE = "percentile"

class TreatmentMethod(Enum):
    """Available outlier treatment methods"""
    REMOVE = "remove"
    CAP_IQR = "cap_iqr"
    CAP_PERCENTILE = "cap_percentile"
    CAP_Z_SCORE = "cap_z_score"
    WINSORIZE = "winsorize"
    LOG_TRANSFORM = "log_transform"
    SQRT_TRANSFORM = "sqrt_transform"
    REPLACE_MEDIAN = "replace_median"
    REPLACE_MEAN = "replace_mean"

class OutlierProcessor:
    """Advanced outlier detection and treatment processor"""
    
    def __init__(self):
        self.z_threshold = 3.0
        self.modified_z_threshold = 3.5
        self.iqr_multiplier = 1.5
        self.percentile_lower = 1
        self.percentile_upper = 99
        
    async def detect_outliers(self, df: pd.DataFrame, method: str = "z_score", column: str = None, **kwargs) -> Dict[str, Any]:
        """
        Detect outliers using specified method
        """
        try:
            result = {
                "method": method,
                "column": column,
                "outliers_detected": {},
                "outlier_indices": {},
                "statistics": {},
                "summary": {}
            }
            
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if column and column in numeric_cols:
                numeric_cols = [column]
            elif column and column not in numeric_cols:
                raise ValueError(f"Column '{column}' is not numeric")
            
            total_outliers = 0
            
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) == 0:
                    continue
                    
                outlier_indices = []
                
                if method == OutlierMethod.Z_SCORE.value:
                    outlier_indices = self._detect_z_score_outliers(col_data, kwargs.get('threshold', self.z_threshold))
                elif method == OutlierMethod.MODIFIED_Z_SCORE.value:
                    outlier_indices = self._detect_modified_z_score_outliers(col_data, kwargs.get('threshold', self.modified_z_threshold))
                elif method == OutlierMethod.IQR.value:
                    outlier_indices = self._detect_iqr_outliers(col_data, kwargs.get('multiplier', self.iqr_multiplier))
                elif method == OutlierMethod.ISOLATION_FOREST.value:
                    outlier_indices = self._detect_isolation_forest_outliers(col_data, kwargs.get('contamination', 0.1))
                elif method == OutlierMethod.PERCENTILE.value:
                    outlier_indices = self._detect_percentile_outliers(
                        col_data, 
                        kwargs.get('lower', self.percentile_lower),
                        kwargs.get('upper', self.percentile_upper)
                    )
                
                # Store results
                result["outliers_detected"][col] = len(outlier_indices)
                result["outlier_indices"][col] = outlier_indices.tolist() if hasattr(outlier_indices, 'tolist') else list(outlier_indices)
                result["statistics"][col] = self._calculate_column_statistics(col_data, outlier_indices)
                total_outliers += len(outlier_indices)
            
            # Summary statistics
            result["summary"] = {
                "total_outliers": total_outliers,
                "columns_with_outliers": len([col for col, count in result["outliers_detected"].items() if count > 0]),
                "outlier_percentage": (total_outliers / len(df)) * 100 if len(df) > 0 else 0,
                "method_used": method,
                "parameters": kwargs
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {str(e)}")
            raise
    
    def _detect_z_score_outliers(self, data: pd.Series, threshold: float = 3.0) -> np.ndarray:
        """Detect outliers using Z-score method"""
        z_scores = np.abs(stats.zscore(data))
        return data.index[z_scores > threshold].values
    
    def _detect_modified_z_score_outliers(self, data: pd.Series, threshold: float = 3.5) -> np.ndarray:
        """Detect outliers using Modified Z-score method (using median)"""
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad
        return data.index[np.abs(modified_z_scores) > threshold].values
    
    def _detect_iqr_outliers(self, data: pd.Series, multiplier: float = 1.5) -> np.ndarray:
        """Detect outliers using IQR method"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        return data.index[(data < lower_bound) | (data > upper_bound)].values
    
    def _detect_isolation_forest_outliers(self, data: pd.Series, contamination: float = 0.1) -> np.ndarray:
        """Detect outliers using Isolation Forest"""
        try:
            clf = IsolationForest(contamination=contamination, random_state=42)
            outliers = clf.fit_predict(data.values.reshape(-1, 1))
            return data.index[outliers == -1].values
        except Exception:
            # Fallback to IQR if Isolation Forest fails
            return self._detect_iqr_outliers(data)
    
    def _detect_percentile_outliers(self, data: pd.Series, lower_percentile: float = 1, upper_percentile: float = 99) -> np.ndarray:
        """Detect outliers using percentile method"""
        lower_bound = np.percentile(data, lower_percentile)
        upper_bound = np.percentile(data, upper_percentile)
        return data.index[(data < lower_bound) | (data > upper_bound)].values
    
    def _calculate_column_statistics(self, data: pd.Series, outlier_indices: np.ndarray) -> Dict[str, Any]:
        """Calculate statistics for a column including outlier information"""
        outlier_values = data.loc[outlier_indices] if len(outlier_indices) > 0 else pd.Series([], dtype=float)
        clean_data = data.drop(outlier_indices) if len(outlier_indices) > 0 else data
        
        return {
            "total_values": len(data),
            "outlier_count": len(outlier_indices),
            "outlier_percentage": (len(outlier_indices) / len(data)) * 100,
            "outlier_values": {
                "min": float(outlier_values.min()) if len(outlier_values) > 0 else None,
                "max": float(outlier_values.max()) if len(outlier_values) > 0 else None,
                "mean": float(outlier_values.mean()) if len(outlier_values) > 0 else None,
                "std": float(outlier_values.std()) if len(outlier_values) > 0 else None
            },
            "clean_data_stats": {
                "min": float(clean_data.min()) if len(clean_data) > 0 else None,
                "max": float(clean_data.max()) if len(clean_data) > 0 else None,
                "mean": float(clean_data.mean()) if len(clean_data) > 0 else None,
                "std": float(clean_data.std()) if len(clean_data) > 0 else None,
                "q1": float(clean_data.quantile(0.25)) if len(clean_data) > 0 else None,
                "q3": float(clean_data.quantile(0.75)) if len(clean_data) > 0 else None
            },
            "original_stats": {
                "min": float(data.min()),
                "max": float(data.max()),
                "mean": float(data.mean()),
                "std": float(data.std()),
                "q1": float(data.quantile(0.25)),
                "q3": float(data.quantile(0.75))
            }
        }
    
    async def analyze_outlier_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive outlier analysis using multiple methods
        """
        try:
            analysis = {
                "dataset_info": {
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "numeric_columns": len(df.select_dtypes(include=[np.number]).columns)
                },
                "detection_methods": {},
                "column_analysis": {},
                "recommendations": [],
                "summary": {}
            }
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Run multiple detection methods
            methods = [
                ("z_score", {"threshold": 3.0}),
                ("modified_z_score", {"threshold": 3.5}),
                ("iqr", {"multiplier": 1.5}),
                ("percentile", {"lower": 1, "upper": 99})
            ]
            
            for method, params in methods:
                try:
                    detection_result = await self.detect_outliers(df, method, **params)
                    analysis["detection_methods"][method] = detection_result["summary"]
                except Exception as e:
                    logger.warning(f"Failed to run {method} detection: {str(e)}")
                    continue
            
            # Detailed column analysis
            for col in numeric_cols:
                col_analysis = {
                    "column_name": col,
                    "data_type": str(df[col].dtype),
                    "total_values": len(df[col].dropna()),
                    "methods_results": {},
                    "consensus_outliers": [],
                    "recommended_treatment": []
                }
                
                # Run each method for this column
                outlier_indices_by_method = {}
                for method, params in methods:
                    try:
                        result = await self.detect_outliers(df, method, col, **params)
                        if col in result["outlier_indices"]:
                            outlier_indices_by_method[method] = set(result["outlier_indices"][col])
                            col_analysis["methods_results"][method] = {
                                "outlier_count": result["outliers_detected"][col],
                                "outlier_percentage": (result["outliers_detected"][col] / len(df[col].dropna())) * 100,
                                "statistics": result["statistics"][col] if col in result["statistics"] else {}
                            }
                    except Exception:
                        continue
                
                # Find consensus outliers (detected by multiple methods)
                if outlier_indices_by_method:
                    all_indices = set()
                    for indices in outlier_indices_by_method.values():
                        all_indices.update(indices)
                    
                    # Count how many methods detected each index as outlier
                    consensus_count = {}
                    for idx in all_indices:
                        count = sum(1 for indices in outlier_indices_by_method.values() if idx in indices)
                        consensus_count[idx] = count
                    
                    # Consider outliers detected by at least 2 methods as consensus
                    col_analysis["consensus_outliers"] = [
                        idx for idx, count in consensus_count.items() if count >= 2
                    ]
                
                # Generate recommendations
                col_analysis["recommended_treatment"] = self._generate_treatment_recommendations(
                    df[col], col_analysis
                )
                
                analysis["column_analysis"][col] = col_analysis
            
            # Generate global recommendations
            analysis["recommendations"] = self._generate_global_recommendations(analysis)
            
            # Summary
            total_consensus_outliers = sum(
                len(col_data["consensus_outliers"]) 
                for col_data in analysis["column_analysis"].values()
            )
            
            analysis["summary"] = {
                "total_consensus_outliers": total_consensus_outliers,
                "columns_with_outliers": len([
                    col for col, data in analysis["column_analysis"].items() 
                    if len(data["consensus_outliers"]) > 0
                ]),
                "consensus_outlier_percentage": (total_consensus_outliers / len(df)) * 100 if len(df) > 0 else 0
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing outlier patterns: {str(e)}")
            raise
    
    def _generate_treatment_recommendations(self, series: pd.Series, col_analysis: Dict) -> List[Dict[str, Any]]:
        """Generate treatment recommendations for a column"""
        recommendations = []
        outlier_count = len(col_analysis["consensus_outliers"])
        total_count = len(series.dropna())
        outlier_percentage = (outlier_count / total_count) * 100 if total_count > 0 else 0
        
        if outlier_percentage <= 1:
            recommendations.append({
                "method": TreatmentMethod.REMOVE.value,
                "priority": 1,
                "reason": f"Very low outlier percentage ({outlier_percentage:.1f}%) - safe to remove",
                "pros": ["Preserves data distribution", "Simple approach"],
                "cons": ["Reduces sample size slightly"]
            })
        
        if outlier_percentage <= 5:
            recommendations.append({
                "method": TreatmentMethod.CAP_IQR.value,
                "priority": 2,
                "reason": "IQR capping preserves data while reducing extreme values",
                "pros": ["Maintains sample size", "Reduces skewness"],
                "cons": ["May lose some information"]
            })
        
        if outlier_percentage > 5:
            recommendations.append({
                "method": TreatmentMethod.WINSORIZE.value,
                "priority": 1,
                "reason": f"High outlier percentage ({outlier_percentage:.1f}%) - winsorizing is safer",
                "pros": ["Maintains sample size", "Reduces impact of extremes"],
                "cons": ["Changes data distribution"]
            })
        
        # Check for positive skewness (log transform might help)
        if series.dropna().skew() > 1:
            recommendations.append({
                "method": TreatmentMethod.LOG_TRANSFORM.value,
                "priority": 3,
                "reason": "Positive skewness detected - log transform can help",
                "pros": ["Reduces skewness", "Handles multiplicative relationships"],
                "cons": ["Changes interpretation", "Requires positive values"]
            })
        
        return recommendations
    
    def _generate_global_recommendations(self, analysis: Dict) -> List[str]:
        """Generate global recommendations for outlier treatment"""
        recommendations = []
        
        total_outliers = analysis["summary"]["total_consensus_outliers"]
        outlier_percentage = analysis["summary"]["consensus_outlier_percentage"]
        
        if outlier_percentage < 1:
            recommendations.append("✅ Very low outlier percentage - consider simple removal")
        elif outlier_percentage < 5:
            recommendations.append("⚠️ Moderate outliers - use capping or winsorizing")
        else:
            recommendations.append("🚨 High outlier percentage - investigate data quality")
        
        # Check for columns with many outliers
        high_outlier_cols = [
            col for col, data in analysis["column_analysis"].items()
            if len(data["consensus_outliers"]) / analysis["dataset_info"]["total_rows"] > 0.1
        ]
        
        if high_outlier_cols:
            recommendations.append(f"Consider data transformation for: {', '.join(high_outlier_cols[:3])}")
        
        recommendations.append("Use multiple detection methods to confirm outliers")
        recommendations.append("Always visualize data before and after treatment")
        
        return recommendations
    
    async def treat_outliers(self, df: pd.DataFrame, method: str, column: str = None, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply outlier treatment method
        """
        try:
            original_shape = df.shape
            treatment_info = {
                "method": method,
                "column": column,
                "original_shape": original_shape,
                "parameters": kwargs
            }
            
            df_treated = df.copy()
            
            # Get columns to treat
            if column:
                columns_to_treat = [column] if column in df.select_dtypes(include=[np.number]).columns else []
            else:
                columns_to_treat = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not columns_to_treat:
                raise ValueError("No numeric columns found to treat")
            
            outliers_removed = 0
            outliers_treated = 0
            
            for col in columns_to_treat:
                col_data = df[col].dropna()
                if len(col_data) == 0:
                    continue
                
                if method == TreatmentMethod.REMOVE.value:
                    # Detect outliers first
                    detection_method = kwargs.get('detection_method', 'iqr')
                    outlier_result = await self.detect_outliers(df, detection_method, col)
                    outlier_indices = outlier_result["outlier_indices"].get(col, [])
                    
                    # Remove rows with outliers
                    df_treated = df_treated.drop(outlier_indices)
                    outliers_removed += len(outlier_indices)
                
                elif method == TreatmentMethod.CAP_IQR.value:
                    Q1 = df_treated[col].quantile(0.25)
                    Q3 = df_treated[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers_mask = (df_treated[col] < lower_bound) | (df_treated[col] > upper_bound)
                    outliers_treated += outliers_mask.sum()
                    
                    df_treated[col] = df_treated[col].clip(lower=lower_bound, upper=upper_bound)
                
                elif method == TreatmentMethod.CAP_PERCENTILE.value:
                    lower_pct = kwargs.get('lower_percentile', 1)
                    upper_pct = kwargs.get('upper_percentile', 99)
                    
                    lower_bound = df_treated[col].quantile(lower_pct / 100)
                    upper_bound = df_treated[col].quantile(upper_pct / 100)
                    
                    outliers_mask = (df_treated[col] < lower_bound) | (df_treated[col] > upper_bound)
                    outliers_treated += outliers_mask.sum()
                    
                    df_treated[col] = df_treated[col].clip(lower=lower_bound, upper=upper_bound)
                
                elif method == TreatmentMethod.CAP_Z_SCORE.value:
                    threshold = kwargs.get('z_threshold', 3.0)
                    mean = df_treated[col].mean()
                    std = df_treated[col].std()
                    
                    lower_bound = mean - threshold * std
                    upper_bound = mean + threshold * std
                    
                    outliers_mask = (df_treated[col] < lower_bound) | (df_treated[col] > upper_bound)
                    outliers_treated += outliers_mask.sum()
                    
                    df_treated[col] = df_treated[col].clip(lower=lower_bound, upper=upper_bound)
                
                elif method == TreatmentMethod.WINSORIZE.value:
                    lower_pct = kwargs.get('lower_percentile', 5)
                    upper_pct = kwargs.get('upper_percentile', 95)
                    
                    lower_bound = df_treated[col].quantile(lower_pct / 100)
                    upper_bound = df_treated[col].quantile(upper_pct / 100)
                    
                    outliers_mask = (df_treated[col] < lower_bound) | (df_treated[col] > upper_bound)
                    outliers_treated += outliers_mask.sum()
                    
                    df_treated[col] = df_treated[col].clip(lower=lower_bound, upper=upper_bound)
                
                elif method == TreatmentMethod.LOG_TRANSFORM.value:
                    if (df_treated[col] <= 0).any():
                        # Add constant to make all values positive
                        min_val = df_treated[col].min()
                        df_treated[col] = df_treated[col] - min_val + 1
                    
                    df_treated[col] = np.log(df_treated[col])
                    outliers_treated += len(df_treated[col])
                
                elif method == TreatmentMethod.SQRT_TRANSFORM.value:
                    if (df_treated[col] < 0).any():
                        # Handle negative values
                        min_val = df_treated[col].min()
                        df_treated[col] = df_treated[col] - min_val
                    
                    df_treated[col] = np.sqrt(df_treated[col])
                    outliers_treated += len(df_treated[col])
                
                elif method == TreatmentMethod.REPLACE_MEDIAN.value:
                    # Detect outliers and replace with median
                    detection_method = kwargs.get('detection_method', 'iqr')
                    outlier_result = await self.detect_outliers(df, detection_method, col)
                    outlier_indices = outlier_result["outlier_indices"].get(col, [])
                    
                    median_val = df_treated[col].median()
                    df_treated.loc[outlier_indices, col] = median_val
                    outliers_treated += len(outlier_indices)
                
                elif method == TreatmentMethod.REPLACE_MEAN.value:
                    # Detect outliers and replace with mean
                    detection_method = kwargs.get('detection_method', 'iqr')
                    outlier_result = await self.detect_outliers(df, detection_method, col)
                    outlier_indices = outlier_result["outlier_indices"].get(col, [])
                    
                    mean_val = df_treated[col].mean()
                    df_treated.loc[outlier_indices, col] = mean_val
                    outliers_treated += len(outlier_indices)
            
            # Update treatment info
            treatment_info["final_shape"] = df_treated.shape
            treatment_info["rows_removed"] = original_shape[0] - df_treated.shape[0]
            treatment_info["outliers_removed"] = outliers_removed
            treatment_info["outliers_treated"] = outliers_treated
            treatment_info["columns_treated"] = columns_to_treat
            
            return df_treated, treatment_info
            
        except Exception as e:
            logger.error(f"Error treating outliers: {str(e)}")
            raise
    
    async def get_treatment_preview(self, df: pd.DataFrame, method: str, column: str = None, **kwargs) -> Dict[str, Any]:
        """
        Get a preview of what the treatment would do without actually applying it
        """
        try:
            preview = {
                "method": method,
                "column": column,
                "current_shape": df.shape,
                "parameters": kwargs
            }
            
            # Get columns to analyze
            if column:
                columns_to_analyze = [column] if column in df.select_dtypes(include=[np.number]).columns else []
            else:
                columns_to_analyze = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not columns_to_analyze:
                preview["error"] = "No numeric columns found"
                return preview
            
            estimated_changes = {}
            
            for col in columns_to_analyze:
                col_data = df[col].dropna()
                if len(col_data) == 0:
                    continue
                
                col_preview = {
                    "current_outliers": 0,
                    "estimated_impact": "",
                    "bounds": {}
                }
                
                if method in [TreatmentMethod.REMOVE.value, TreatmentMethod.REPLACE_MEDIAN.value, TreatmentMethod.REPLACE_MEAN.value]:
                    # Detect outliers to estimate removal/replacement
                    detection_method = kwargs.get('detection_method', 'iqr')
                    try:
                        outlier_result = await self.detect_outliers(df, detection_method, col)
                        outlier_count = outlier_result["outliers_detected"].get(col, 0)
                        col_preview["current_outliers"] = outlier_count
                        
                        if method == TreatmentMethod.REMOVE.value:
                            col_preview["estimated_impact"] = f"{outlier_count} rows would be removed"
                        else:
                            replacement_val = "median" if method == TreatmentMethod.REPLACE_MEDIAN.value else "mean"
                            col_preview["estimated_impact"] = f"{outlier_count} outliers would be replaced with {replacement_val}"
                    except:
                        col_preview["estimated_impact"] = "Unable to estimate"
                
                elif method == TreatmentMethod.CAP_IQR.value:
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
                    col_preview["current_outliers"] = outliers
                    col_preview["bounds"] = {"lower": lower_bound, "upper": upper_bound}
                    col_preview["estimated_impact"] = f"{outliers} values would be capped to [{lower_bound:.2f}, {upper_bound:.2f}]"
                
                elif method == TreatmentMethod.CAP_PERCENTILE.value:
                    lower_pct = kwargs.get('lower_percentile', 1)
                    upper_pct = kwargs.get('upper_percentile', 99)
                    
                    lower_bound = col_data.quantile(lower_pct / 100)
                    upper_bound = col_data.quantile(upper_pct / 100)
                    
                    outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
                    col_preview["current_outliers"] = outliers
                    col_preview["bounds"] = {"lower": lower_bound, "upper": upper_bound}
                    col_preview["estimated_impact"] = f"{outliers} values would be capped to [{lower_bound:.2f}, {upper_bound:.2f}]"
                
                estimated_changes[col] = col_preview
            
            preview["estimated_changes"] = estimated_changes
            
            # Estimate final shape
            if method == TreatmentMethod.REMOVE.value:
                total_rows_to_remove = sum(
                    changes["current_outliers"] 
                    for changes in estimated_changes.values()
                )
                preview["estimated_final_shape"] = (df.shape[0] - total_rows_to_remove, df.shape[1])
            else:
                preview["estimated_final_shape"] = df.shape
            
            return preview
            
        except Exception as e:
            logger.error(f"Error generating treatment preview: {str(e)}")
            raise
