"""
Advanced Missing Values Analysis and Treatment Processor
Provides comprehensive missing value analysis, pattern detection, and multiple treatment methods
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from enum import Enum
import logging
from scipy import stats
from sklearn.impute import SimpleImputer
try:
    from sklearn.impute import KNNImputer
    HAS_KNN = True
except ImportError:
    HAS_KNN = False

try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    HAS_ITERATIVE = True
except ImportError:
    HAS_ITERATIVE = False
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class MissingPattern(Enum):
    """Types of missing data patterns"""
    MCAR = "Missing Completely At Random"  # Missing values are random
    MAR = "Missing At Random"  # Missing depends on observed data
    MNAR = "Missing Not At Random"  # Missing depends on unobserved data
    STRUCTURAL = "Structural Missing"  # Missing by design

class TreatmentMethod(Enum):
    """Available missing value treatment methods"""
    DROP_ROWS = "drop_rows"
    DROP_COLUMNS = "drop_columns"
    MEAN_IMPUTATION = "mean_imputation"
    MEDIAN_IMPUTATION = "median_imputation"
    MODE_IMPUTATION = "mode_imputation"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    KNN_IMPUTATION = "knn_imputation"
    ITERATIVE_IMPUTATION = "iterative_imputation"
    CONSTANT_IMPUTATION = "constant_imputation"

class MissingValuesProcessor:
    """Advanced missing values analysis and treatment processor"""
    
    def __init__(self):
        self.missing_threshold_drop = 0.05  # 5% threshold for dropping rows
        self.column_threshold_drop = 0.50   # 50% threshold for dropping columns
        
    async def analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive analysis of missing value patterns
        """
        try:
            analysis = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "missing_summary": {},
                "column_analysis": {},
                "pattern_analysis": {},
                "recommendations": []
            }
            
            # Basic missing value statistics
            missing_counts = df.isnull().sum()
            missing_percentages = (missing_counts / len(df)) * 100
            
            analysis["missing_summary"] = {
                "total_missing_values": int(missing_counts.sum()),
                "columns_with_missing": int((missing_counts > 0).sum()),
                "percentage_missing_overall": float(missing_counts.sum() / (len(df) * len(df.columns)) * 100)
            }
            
            # Per-column analysis
            for col in df.columns:
                col_missing = missing_counts[col]
                col_percentage = missing_percentages[col]
                
                if col_missing > 0:
                    col_analysis = {
                        "missing_count": int(col_missing),
                        "missing_percentage": float(col_percentage),
                        "data_type": str(df[col].dtype),
                        "unique_values": int(df[col].nunique()),
                        "pattern_type": self._detect_missing_pattern(df, col),
                        "reasons": self._analyze_missing_reasons(df, col),
                        "recommended_methods": self._recommend_treatment_methods(df, col, col_percentage)
                    }
                    analysis["column_analysis"][col] = col_analysis
            
            # Pattern analysis
            analysis["pattern_analysis"] = self._analyze_missing_patterns(df)
            
            # Global recommendations
            analysis["recommendations"] = self._generate_global_recommendations(df, analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing missing patterns: {str(e)}")
            raise
    
    def _detect_missing_pattern(self, df: pd.DataFrame, column: str) -> str:
        """Detect the type of missing pattern for a column"""
        try:
            missing_mask = df[column].isnull()
            
            # Check if missing values are completely random
            if self._test_mcar(df, column):
                return MissingPattern.MCAR.value
            
            # Check if missing values depend on other observed variables
            if self._test_mar(df, column):
                return MissingPattern.MAR.value
            
            # Check for structural patterns
            if self._test_structural(df, column):
                return MissingPattern.STRUCTURAL.value
            
            # Default to MNAR if no clear pattern
            return MissingPattern.MNAR.value
            
        except Exception:
            return "Unknown"
    
    def _test_mcar(self, df: pd.DataFrame, column: str) -> bool:
        """Test if missing values are Missing Completely At Random"""
        try:
            missing_mask = df[column].isnull()
            
            # Test correlation with other variables
            correlations = []
            for other_col in df.columns:
                if other_col != column and df[other_col].dtype in ['int64', 'float64']:
                    try:
                        # Calculate point-biserial correlation
                        corr, p_value = stats.pointbiserialr(missing_mask, df[other_col].fillna(df[other_col].mean()))
                        if p_value < 0.05 and abs(corr) > 0.1:
                            correlations.append(abs(corr))
                    except:
                        continue
            
            # If no significant correlations, likely MCAR
            return len(correlations) == 0 or max(correlations, default=0) < 0.1
            
        except Exception:
            return False
    
    def _test_mar(self, df: pd.DataFrame, column: str) -> bool:
        """Test if missing values are Missing At Random"""
        try:
            missing_mask = df[column].isnull()
            
            # Check if missingness depends on other observed variables
            for other_col in df.columns:
                if other_col != column:
                    try:
                        if df[other_col].dtype == 'object':
                            # Chi-square test for categorical variables
                            contingency = pd.crosstab(missing_mask, df[other_col].fillna('Unknown'))
                            chi2, p_value = stats.chi2_contingency(contingency)[:2]
                            if p_value < 0.05:
                                return True
                        else:
                            # T-test for numerical variables
                            present_values = df[~missing_mask][other_col].dropna()
                            missing_values = df[missing_mask][other_col].dropna()
                            if len(present_values) > 0 and len(missing_values) > 0:
                                t_stat, p_value = stats.ttest_ind(present_values, missing_values)
                                if p_value < 0.05:
                                    return True
                    except:
                        continue
            
            return False
            
        except Exception:
            return False
    
    def _test_structural(self, df: pd.DataFrame, column: str) -> bool:
        """Test if missing values are structural (by design)"""
        try:
            missing_mask = df[column].isnull()
            missing_percentage = missing_mask.sum() / len(df)
            
            # Check for patterns that suggest structural missingness
            # High percentage of missing values in specific patterns
            if missing_percentage > 0.8:
                return True
            
            # Check if missing values follow a specific pattern (e.g., every nth row)
            missing_indices = df[missing_mask].index.tolist()
            if len(missing_indices) > 1:
                differences = np.diff(missing_indices)
                if len(set(differences)) == 1:  # Regular pattern
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _analyze_missing_reasons(self, df: pd.DataFrame, column: str) -> List[str]:
        """Analyze potential reasons for missing values"""
        reasons = []
        missing_mask = df[column].isnull()
        missing_percentage = missing_mask.sum() / len(df) * 100
        
        # Data collection issues
        if missing_percentage > 50:
            reasons.append("High missing percentage suggests systematic data collection issues")
        
        # Sensor/measurement failures
        if df[column].dtype in ['int64', 'float64']:
            reasons.append("Possible sensor malfunction or measurement device failure")
        
        # User input issues
        if df[column].dtype == 'object':
            reasons.append("Possible user input errors or optional field not filled")
        
        # Temporal patterns
        if 'time' in column.lower() or 'date' in column.lower():
            reasons.append("Temporal data gaps due to system downtime or maintenance")
        
        # Correlation with other missing values
        other_missing = df.isnull().sum()
        correlated_missing = []
        for other_col in df.columns:
            if other_col != column and other_missing[other_col] > 0:
                # Check if missing patterns are similar
                correlation = missing_mask.corr(df[other_col].isnull())
                if correlation > 0.7:
                    correlated_missing.append(other_col)
        
        if correlated_missing:
            reasons.append(f"Missing values correlated with: {', '.join(correlated_missing[:3])}")
        
        # Default reason if none found
        if not reasons:
            reasons.append("Missing values appear to be random or due to unknown factors")
        
        return reasons
    
    def _recommend_treatment_methods(self, df: pd.DataFrame, column: str, missing_percentage: float) -> List[Dict[str, Any]]:
        """Recommend appropriate treatment methods for a column"""
        recommendations = []
        
        # Drop rows if missing percentage is very low
        if missing_percentage <= 5:
            recommendations.append({
                "method": TreatmentMethod.DROP_ROWS.value,
                "priority": 1,
                "reason": f"Only {missing_percentage:.1f}% missing - safe to drop rows",
                "pros": ["Preserves data integrity", "No imputation bias"],
                "cons": ["Reduces sample size slightly"]
            })
        
        # Drop column if missing percentage is very high
        if missing_percentage >= 70:
            recommendations.append({
                "method": TreatmentMethod.DROP_COLUMNS.value,
                "priority": 1,
                "reason": f"{missing_percentage:.1f}% missing - column may not be useful",
                "pros": ["Removes unreliable data", "Simplifies analysis"],
                "cons": ["Loses potentially valuable information"]
            })
        
        # Numerical columns
        if df[column].dtype in ['int64', 'float64']:
            # Mean imputation
            recommendations.append({
                "method": TreatmentMethod.MEAN_IMPUTATION.value,
                "priority": 2,
                "reason": "Suitable for normally distributed numerical data",
                "pros": ["Simple and fast", "Preserves mean"],
                "cons": ["Reduces variance", "May not work well with skewed data"]
            })
            
            # Median imputation
            recommendations.append({
                "method": TreatmentMethod.MEDIAN_IMPUTATION.value,
                "priority": 2,
                "reason": "Better for skewed numerical data",
                "pros": ["Robust to outliers", "Works well with skewed data"],
                "cons": ["May not preserve relationships"]
            })
            
            # KNN imputation
            if missing_percentage < 30:
                recommendations.append({
                    "method": TreatmentMethod.KNN_IMPUTATION.value,
                    "priority": 3,
                    "reason": "Uses similar records for imputation",
                    "pros": ["Considers relationships", "More accurate"],
                    "cons": ["Computationally expensive", "Sensitive to outliers"]
                })
        
        # Categorical columns
        elif df[column].dtype == 'object':
            recommendations.append({
                "method": TreatmentMethod.MODE_IMPUTATION.value,
                "priority": 2,
                "reason": "Most frequent value for categorical data",
                "pros": ["Preserves distribution", "Simple approach"],
                "cons": ["May introduce bias", "Ignores relationships"]
            })
        
        # Time series data
        if any(keyword in column.lower() for keyword in ['time', 'date', 'timestamp']):
            recommendations.append({
                "method": TreatmentMethod.FORWARD_FILL.value,
                "priority": 2,
                "reason": "Forward fill for time series data",
                "pros": ["Maintains temporal continuity"],
                "cons": ["May propagate errors"]
            })
        
        return recommendations
    
    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall missing data patterns"""
        missing_matrix = df.isnull()
        
        # Pattern combinations
        pattern_counts = missing_matrix.value_counts()
        
        # Most common missing patterns
        common_patterns = []
        for pattern, count in pattern_counts.head(5).items():
            if any(pattern):  # Only patterns with missing values
                missing_cols = [col for col, is_missing in zip(df.columns, pattern) if is_missing]
                common_patterns.append({
                    "columns": missing_cols,
                    "count": int(count),
                    "percentage": float(count / len(df) * 100)
                })
        
        return {
            "total_patterns": len(pattern_counts),
            "complete_cases": int((~missing_matrix.any(axis=1)).sum()),
            "complete_cases_percentage": float((~missing_matrix.any(axis=1)).sum() / len(df) * 100),
            "common_patterns": common_patterns
        }
    
    def _generate_global_recommendations(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> List[str]:
        """Generate global recommendations for missing value treatment"""
        recommendations = []
        
        missing_percentage = analysis["missing_summary"]["percentage_missing_overall"]
        
        if missing_percentage < 1:
            recommendations.append("✅ Very low missing data - consider simple imputation or row deletion")
        elif missing_percentage < 5:
            recommendations.append("✅ Low missing data - multiple treatment options available")
        elif missing_percentage < 20:
            recommendations.append("⚠️ Moderate missing data - careful method selection required")
        else:
            recommendations.append("🚨 High missing data - consider data collection review")
        
        # Check for columns with high missing percentages
        high_missing_cols = [col for col, info in analysis["column_analysis"].items() 
                           if info["missing_percentage"] > 50]
        if high_missing_cols:
            recommendations.append(f"Consider dropping columns: {', '.join(high_missing_cols[:3])}")
        
        # Check for correlated missing patterns
        if analysis["pattern_analysis"]["complete_cases_percentage"] < 50:
            recommendations.append("Multiple columns have missing values - consider multivariate imputation")
        
        return recommendations
    
    async def treat_missing_values(self, df: pd.DataFrame, method: str, column: str = None, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply specified missing value treatment method
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
            
            if method == TreatmentMethod.DROP_ROWS.value:
                if column:
                    df_treated = df_treated.dropna(subset=[column])
                else:
                    df_treated = df_treated.dropna()
                    
            elif method == TreatmentMethod.DROP_COLUMNS.value:
                if column:
                    df_treated = df_treated.drop(columns=[column])
                else:
                    # Drop columns with high missing percentage
                    threshold = kwargs.get('threshold', 0.5)
                    missing_pct = df.isnull().sum() / len(df)
                    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
                    df_treated = df_treated.drop(columns=cols_to_drop)
                    treatment_info["dropped_columns"] = cols_to_drop
                    
            elif method == TreatmentMethod.MEAN_IMPUTATION.value:
                if column:
                    mean_value = df[column].mean()
                    df_treated[column] = df_treated[column].fillna(mean_value)
                    treatment_info["imputation_value"] = mean_value
                else:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        mean_value = df[col].mean()
                        df_treated[col] = df_treated[col].fillna(mean_value)
                        
            elif method == TreatmentMethod.MEDIAN_IMPUTATION.value:
                if column:
                    median_value = df[column].median()
                    df_treated[column] = df_treated[column].fillna(median_value)
                    treatment_info["imputation_value"] = median_value
                else:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        median_value = df[col].median()
                        df_treated[col] = df_treated[col].fillna(median_value)
                        
            elif method == TreatmentMethod.MODE_IMPUTATION.value:
                if column:
                    mode_value = df[column].mode().iloc[0] if not df[column].mode().empty else 'Unknown'
                    df_treated[column] = df_treated[column].fillna(mode_value)
                    treatment_info["imputation_value"] = mode_value
                else:
                    categorical_cols = df.select_dtypes(include=['object']).columns
                    for col in categorical_cols:
                        mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                        df_treated[col] = df_treated[col].fillna(mode_value)
                        
            elif method == TreatmentMethod.FORWARD_FILL.value:
                if column:
                    df_treated[column] = df_treated[column].ffill()
                else:
                    df_treated = df_treated.ffill()
                    
            elif method == TreatmentMethod.BACKWARD_FILL.value:
                if column:
                    df_treated[column] = df_treated[column].bfill()
                else:
                    df_treated = df_treated.bfill()
                    
            elif method == TreatmentMethod.KNN_IMPUTATION.value:
                if not HAS_KNN:
                    raise ValueError("KNN imputation not available. Please install scikit-learn >= 0.22")
                n_neighbors = kwargs.get('n_neighbors', 5)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    imputer = KNNImputer(n_neighbors=n_neighbors)
                    df_treated[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                    treatment_info["n_neighbors"] = n_neighbors
                    
            elif method == TreatmentMethod.ITERATIVE_IMPUTATION.value:
                if not HAS_ITERATIVE:
                    raise ValueError("Iterative imputation not available. Please install scikit-learn >= 0.21")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    imputer = IterativeImputer(random_state=42)
                    df_treated[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                    
            elif method == TreatmentMethod.CONSTANT_IMPUTATION.value:
                constant_value = kwargs.get('constant_value', 0)
                if column:
                    df_treated[column] = df_treated[column].fillna(constant_value)
                    treatment_info["constant_value"] = constant_value
                else:
                    df_treated = df_treated.fillna(constant_value)
                    treatment_info["constant_value"] = constant_value
            
            # Update treatment info
            treatment_info["final_shape"] = df_treated.shape
            treatment_info["rows_affected"] = original_shape[0] - df_treated.shape[0]
            treatment_info["missing_values_remaining"] = int(df_treated.isnull().sum().sum())
            
            return df_treated, treatment_info
            
        except Exception as e:
            logger.error(f"Error treating missing values: {str(e)}")
            raise
    
    async def get_treatment_preview(self, df: pd.DataFrame, method: str, column: str = None, **kwargs) -> Dict[str, Any]:
        """
        Get a preview of what the treatment would do without actually applying it
        """
        try:
            preview = {
                "method": method,
                "column": column,
                "current_missing": int(df.isnull().sum().sum()),
                "current_shape": df.shape
            }
            
            if method == TreatmentMethod.DROP_ROWS.value:
                if column:
                    rows_to_drop = df[column].isnull().sum()
                else:
                    rows_to_drop = df.isnull().any(axis=1).sum()
                preview["rows_to_remove"] = int(rows_to_drop)
                preview["final_shape"] = (df.shape[0] - rows_to_drop, df.shape[1])
                
            elif method == TreatmentMethod.DROP_COLUMNS.value:
                if column:
                    preview["columns_to_remove"] = [column]
                    preview["final_shape"] = (df.shape[0], df.shape[1] - 1)
                else:
                    threshold = kwargs.get('threshold', 0.5)
                    missing_pct = df.isnull().sum() / len(df)
                    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
                    preview["columns_to_remove"] = cols_to_drop
                    preview["final_shape"] = (df.shape[0], df.shape[1] - len(cols_to_drop))
                    
            else:
                # For imputation methods, shape stays the same
                preview["final_shape"] = df.shape
                preview["missing_after_treatment"] = 0
                
            return preview
            
        except Exception as e:
            logger.error(f"Error generating treatment preview: {str(e)}")
            raise
