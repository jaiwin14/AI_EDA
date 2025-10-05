"""
Advanced Missing Values Analysis and Treatment Processor - FIXED VERSION
Provides comprehensive missing value analysis, pattern detection, and multiple treatment methods
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from enum import Enum
import logging
from scipy import stats
from sklearn.impute import SimpleImputer

# Safe imports with fallbacks
try:
    from sklearn.impute import KNNImputer
    HAS_KNN = True
except ImportError:
    HAS_KNN = False
    print("⚠️ KNNImputer not available. Install scikit-learn >= 0.22 for KNN imputation.")

try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    HAS_ITERATIVE = True
except ImportError:
    HAS_ITERATIVE = False
    print("⚠️ IterativeImputer not available. Install scikit-learn >= 0.21 for iterative imputation.")

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MissingPattern(Enum):
    """Types of missing data patterns"""
    MCAR = "Missing Completely At Random"
    MAR = "Missing At Random"
    MNAR = "Missing Not At Random"
    STRUCTURAL = "Structural Missing"


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
    """Advanced missing values analysis and treatment processor - FIXED"""
    
    def __init__(self):
        self.missing_threshold_drop = 0.05  # 5% threshold for dropping rows
        self.column_threshold_drop = 0.50   # 50% threshold for dropping columns
        
    async def analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive analysis of missing value patterns - FIXED
        """
        try:
            logger.info(f"📊 Starting missing values analysis for dataset with shape {df.shape}")
            
            analysis = {
                "total_rows": int(len(df)),
                "total_columns": int(len(df.columns)),
                "missing_summary": {},
                "column_analysis": {},
                "pattern_analysis": {},
                "recommendations": []
            }
            
            # Basic missing value statistics
            missing_counts = df.isnull().sum()
            missing_percentages = (missing_counts / len(df)) * 100
            
            total_missing = int(missing_counts.sum())
            columns_with_missing = int((missing_counts > 0).sum())
            total_cells = len(df) * len(df.columns)
            overall_missing_pct = float((total_missing / total_cells * 100) if total_cells > 0 else 0)
            
            analysis["missing_summary"] = {
                "total_missing_values": total_missing,
                "columns_with_missing": columns_with_missing,
                "percentage_missing_overall": round(overall_missing_pct, 2)
            }
            
            logger.info(f"📊 Total missing: {total_missing}, Columns affected: {columns_with_missing}")
            
            # Per-column analysis
            for col in df.columns:
                col_missing = int(missing_counts[col])
                col_percentage = float(missing_percentages[col])
                
                if col_missing > 0:
                    col_analysis = {
                        "missing_count": col_missing,
                        "missing_percentage": round(col_percentage, 2),
                        "data_type": str(df[col].dtype),
                        "unique_values": int(df[col].nunique()),
                        "pattern_type": self._detect_missing_pattern(df, col),
                        "reasons": self._analyze_missing_reasons(df, col),
                        "recommended_methods": self._recommend_treatment_methods(df, col, col_percentage)
                    }
                    analysis["column_analysis"][col] = col_analysis
                    logger.info(f"  Column '{col}': {col_missing} missing ({col_percentage:.1f}%)")
            
            # Pattern analysis
            analysis["pattern_analysis"] = self._analyze_missing_patterns(df)
            
            # Global recommendations
            analysis["recommendations"] = self._generate_global_recommendations(df, analysis)
            
            logger.info(f"✅ Analysis complete")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing missing patterns: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
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
                        corr, p_value = stats.pointbiserialr(
                            missing_mask, 
                            df[other_col].fillna(df[other_col].mean())
                        )
                        if p_value < 0.05 and abs(corr) > 0.1:
                            correlations.append(abs(corr))
                    except:
                        continue
            
            return len(correlations) == 0 or max(correlations, default=0) < 0.1
            
        except Exception:
            return False
    
    def _test_mar(self, df: pd.DataFrame, column: str) -> bool:
        """Test if missing values are Missing At Random"""
        try:
            missing_mask = df[column].isnull()
            
            for other_col in df.columns:
                if other_col != column:
                    try:
                        if df[other_col].dtype == 'object':
                            contingency = pd.crosstab(missing_mask, df[other_col].fillna('Unknown'))
                            chi2, p_value = stats.chi2_contingency(contingency)[:2]
                            if p_value < 0.05:
                                return True
                        else:
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
            
            if missing_percentage > 0.8:
                return True
            
            missing_indices = df[missing_mask].index.tolist()
            if len(missing_indices) > 1:
                differences = np.diff(missing_indices)
                if len(set(differences)) == 1:
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _analyze_missing_reasons(self, df: pd.DataFrame, column: str) -> List[str]:
        """Analyze potential reasons for missing values"""
        reasons = []
        missing_mask = df[column].isnull()
        missing_percentage = missing_mask.sum() / len(df) * 100
        
        if missing_percentage > 50:
            reasons.append("High missing percentage suggests systematic data collection issues")
        
        if df[column].dtype in ['int64', 'float64']:
            reasons.append("Possible sensor malfunction or measurement device failure")
        
        if df[column].dtype == 'object':
            reasons.append("Possible user input errors or optional field not filled")
        
        if 'time' in column.lower() or 'date' in column.lower():
            reasons.append("Temporal data gaps due to system downtime or maintenance")
        
        other_missing = df.isnull().sum()
        correlated_missing = []
        for other_col in df.columns:
            if other_col != column and other_missing[other_col] > 0:
                correlation = missing_mask.corr(df[other_col].isnull())
                if correlation > 0.7:
                    correlated_missing.append(other_col)
        
        if correlated_missing:
            reasons.append(f"Missing values correlated with: {', '.join(correlated_missing[:3])}")
        
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
            recommendations.append({
                "method": TreatmentMethod.MEAN_IMPUTATION.value,
                "priority": 2,
                "reason": "Suitable for normally distributed numerical data",
                "pros": ["Simple and fast", "Preserves mean"],
                "cons": ["Reduces variance", "May not work well with skewed data"]
            })
            
            recommendations.append({
                "method": TreatmentMethod.MEDIAN_IMPUTATION.value,
                "priority": 2,
                "reason": "Better for skewed numerical data",
                "pros": ["Robust to outliers", "Works well with skewed data"],
                "cons": ["May not preserve relationships"]
            })
            
            if missing_percentage < 30 and HAS_KNN:
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
        
        pattern_counts = missing_matrix.value_counts()
        
        common_patterns = []
        for pattern, count in pattern_counts.head(5).items():
            if any(pattern):
                missing_cols = [col for col, is_missing in zip(df.columns, pattern) if is_missing]
                common_patterns.append({
                    "columns": missing_cols,
                    "count": int(count),
                    "percentage": round(float(count / len(df) * 100), 2)
                })
        
        complete_cases = int((~missing_matrix.any(axis=1)).sum())
        complete_pct = round(float(complete_cases / len(df) * 100), 2)
        
        return {
            "total_patterns": int(len(pattern_counts)),
            "complete_cases": complete_cases,
            "complete_cases_percentage": complete_pct,
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
        
        high_missing_cols = [col for col, info in analysis["column_analysis"].items() 
                           if info["missing_percentage"] > 50]
        if high_missing_cols:
            recommendations.append(f"Consider dropping columns: {', '.join(high_missing_cols[:3])}")
        
        if analysis["pattern_analysis"]["complete_cases_percentage"] < 50:
            recommendations.append("Multiple columns have missing values - consider multivariate imputation")
        
        return recommendations
    
    async def treat_missing_values(
        self, 
        df: pd.DataFrame, 
        method: str, 
        column: Optional[str] = None, 
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply specified missing value treatment method - FIXED
        """
        try:
            logger.info(f"🔧 Applying treatment: {method}")
            logger.info(f"   Column: {column if column else 'All columns'}")
            logger.info(f"   Parameters: {kwargs}")
            
            original_shape = df.shape
            original_missing = int(df.isnull().sum().sum())
            
            treatment_info = {
                "method": method,
                "column": column,
                "original_shape": list(original_shape),
                "parameters": kwargs
            }
            
            df_treated = df.copy()
            
            # DROP ROWS
            if method == TreatmentMethod.DROP_ROWS.value:
                if column:
                    df_treated = df_treated.dropna(subset=[column])
                else:
                    df_treated = df_treated.dropna()
                logger.info(f"   Dropped {original_shape[0] - df_treated.shape[0]} rows")
                    
            # DROP COLUMNS
            elif method == TreatmentMethod.DROP_COLUMNS.value:
                if column:
                    df_treated = df_treated.drop(columns=[column])
                    treatment_info["dropped_columns"] = [column]
                else:
                    threshold = kwargs.get('threshold', 0.5)
                    missing_pct = df.isnull().sum() / len(df)
                    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
                    if cols_to_drop:
                        df_treated = df_treated.drop(columns=cols_to_drop)
                    treatment_info["dropped_columns"] = cols_to_drop
                logger.info(f"   Dropped columns: {treatment_info.get('dropped_columns', [])}")
                    
            # MEAN IMPUTATION
            elif method == TreatmentMethod.MEAN_IMPUTATION.value:
                if column:
                    if df[column].dtype in ['int64', 'float64']:
                        mean_value = float(df[column].mean())
                        df_treated[column] = df_treated[column].fillna(mean_value)
                        treatment_info["imputation_value"] = mean_value
                    else:
                        raise ValueError(f"Cannot apply mean imputation to non-numeric column '{column}'")
                else:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        if df[col].isnull().any():
                            mean_value = float(df[col].mean())
                            df_treated[col] = df_treated[col].fillna(mean_value)
                logger.info(f"   Applied mean imputation")
                        
            # MEDIAN IMPUTATION
            elif method == TreatmentMethod.MEDIAN_IMPUTATION.value:
                if column:
                    if df[column].dtype in ['int64', 'float64']:
                        median_value = float(df[column].median())
                        df_treated[column] = df_treated[column].fillna(median_value)
                        treatment_info["imputation_value"] = median_value
                    else:
                        raise ValueError(f"Cannot apply median imputation to non-numeric column '{column}'")
                else:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        if df[col].isnull().any():
                            median_value = float(df[col].median())
                            df_treated[col] = df_treated[col].fillna(median_value)
                logger.info(f"   Applied median imputation")
                        
            # MODE IMPUTATION
            elif method == TreatmentMethod.MODE_IMPUTATION.value:
                if column:
                    mode_series = df[column].mode()
                    mode_value = mode_series.iloc[0] if len(mode_series) > 0 else 'Unknown'
                    df_treated[column] = df_treated[column].fillna(mode_value)
                    treatment_info["imputation_value"] = str(mode_value)
                else:
                    for col in df.columns:
                        if df[col].isnull().any():
                            mode_series = df[col].mode()
                            mode_value = mode_series.iloc[0] if len(mode_series) > 0 else 'Unknown'
                            df_treated[col] = df_treated[col].fillna(mode_value)
                logger.info(f"   Applied mode imputation")
                        
            # FORWARD FILL
            elif method == TreatmentMethod.FORWARD_FILL.value:
                if column:
                    df_treated[column] = df_treated[column].ffill()
                else:
                    df_treated = df_treated.ffill()
                logger.info(f"   Applied forward fill")
                    
            # BACKWARD FILL
            elif method == TreatmentMethod.BACKWARD_FILL.value:
                if column:
                    df_treated[column] = df_treated[column].bfill()
                else:
                    df_treated = df_treated.bfill()
                logger.info(f"   Applied backward fill")
                    
            # KNN IMPUTATION
            elif method == TreatmentMethod.KNN_IMPUTATION.value:
                if not HAS_KNN:
                    raise ValueError("KNN imputation not available. Please install scikit-learn >= 0.22")
                
                n_neighbors = kwargs.get('n_neighbors', 5)
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_cols) == 0:
                    raise ValueError("No numeric columns found for KNN imputation")
                
                if column and column in numeric_cols:
                    numeric_cols = [column]
                
                imputer = KNNImputer(n_neighbors=n_neighbors)
                df_treated[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                treatment_info["n_neighbors"] = n_neighbors
                logger.info(f"   Applied KNN imputation with {n_neighbors} neighbors")
                    
            # ITERATIVE IMPUTATION
            elif method == TreatmentMethod.ITERATIVE_IMPUTATION.value:
                if not HAS_ITERATIVE:
                    raise ValueError("Iterative imputation not available. Please install scikit-learn >= 0.21")
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_cols) == 0:
                    raise ValueError("No numeric columns found for iterative imputation")
                
                if column and column in numeric_cols:
                    numeric_cols = [column]
                
                imputer = IterativeImputer(random_state=42, max_iter=10)
                df_treated[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                logger.info(f"   Applied iterative imputation")
                    
            # CONSTANT IMPUTATION
            elif method == TreatmentMethod.CONSTANT_IMPUTATION.value:
                constant_value = kwargs.get('constant_value', 0)
                if column:
                    df_treated[column] = df_treated[column].fillna(constant_value)
                    treatment_info["constant_value"] = str(constant_value)
                else:
                    df_treated = df_treated.fillna(constant_value)
                    treatment_info["constant_value"] = str(constant_value)
                logger.info(f"   Applied constant imputation with value: {constant_value}")
            
            else:
                raise ValueError(f"Unknown treatment method: {method}")
            
            # Update treatment info
            final_missing = int(df_treated.isnull().sum().sum())
            treatment_info.update({
                "final_shape": list(df_treated.shape),
                "rows_affected": int(original_shape[0] - df_treated.shape[0]),
                "columns_affected": int(original_shape[1] - df_treated.shape[1]),
                "missing_values_remaining": final_missing,
                "missing_values_removed": original_missing - final_missing
            })
            
            logger.info(f"✅ Treatment complete:")
            logger.info(f"   Original: {original_shape}, Missing: {original_missing}")
            logger.info(f"   Final: {df_treated.shape}, Missing: {final_missing}")
            
            return df_treated, treatment_info
            
        except Exception as e:
            logger.error(f"❌ Error treating missing values: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    async def get_treatment_preview(
        self, 
        df: pd.DataFrame, 
        method: str, 
        column: Optional[str] = None, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get a preview of what the treatment would do - FIXED
        """
        try:
            logger.info(f"👁️ Generating preview for method: {method}")
            
            preview = {
                "method": method,
                "column": column,
                "current_missing": int(df.isnull().sum().sum()),
                "current_shape": list(df.shape)
            }
            
            if method == TreatmentMethod.DROP_ROWS.value:
                if column:
                    rows_to_drop = int(df[column].isnull().sum())
                else:
                    rows_to_drop = int(df.isnull().any(axis=1).sum())
                preview["rows_to_remove"] = rows_to_drop
                preview["final_shape"] = [df.shape[0] - rows_to_drop, df.shape[1]]
                
            elif method == TreatmentMethod.DROP_COLUMNS.value:
                if column:
                    preview["columns_to_remove"] = [column]
                    preview["final_shape"] = [df.shape[0], df.shape[1] - 1]
                else:
                    threshold = kwargs.get('threshold', 0.5)
                    missing_pct = df.isnull().sum() / len(df)
                    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
                    preview["columns_to_remove"] = cols_to_drop
                    preview["final_shape"] = [df.shape[0], df.shape[1] - len(cols_to_drop)]
                    
            else:
                # For imputation methods, shape stays the same
                preview["final_shape"] = list(df.shape)
                preview["missing_after_treatment"] = 0
                preview["rows_to_remove"] = 0
                preview["columns_to_remove"] = []
            
            logger.info(f"✅ Preview generated: {preview}")
            return preview
            
        except Exception as e:
            logger.error(f"❌ Error generating treatment preview: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise