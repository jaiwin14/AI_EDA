"""
Data Validation Utilities
Validate datasets for ML pipeline compatibility
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Comprehensive data validation for ML pipelines"""
    
    def __init__(self):
        self.validation_rules = {
            'min_rows': 10,
            'min_columns': 2,
            'max_missing_percentage': 95,
            'max_duplicate_percentage': 90,
            'min_unique_values': 1
        }
    
    async def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive dataset validation
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validation results dictionary
        """
        
        results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "validation_summary": {}
        }
        
        # Basic structure validation
        structure_results = await self._validate_structure(df)
        results["validation_summary"]["structure"] = structure_results
        
        if not structure_results["is_valid"]:
            results["is_valid"] = False
            results["errors"].extend(structure_results["errors"])
        
        results["warnings"].extend(structure_results["warnings"])
        results["recommendations"].extend(structure_results["recommendations"])
        
        # Data quality validation
        quality_results = await self._validate_data_quality(df)
        results["validation_summary"]["data_quality"] = quality_results
        
        if not quality_results["is_valid"]:
            results["warnings"].extend(quality_results["warnings"])
        
        results["recommendations"].extend(quality_results["recommendations"])
        
        # ML readiness validation
        ml_results = await self._validate_ml_readiness(df)
        results["validation_summary"]["ml_readiness"] = ml_results
        
        results["warnings"].extend(ml_results["warnings"])
        results["recommendations"].extend(ml_results["recommendations"])
        
        return results
    
    async def _validate_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate basic dataset structure"""
        
        results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Check minimum rows
        if len(df) < self.validation_rules['min_rows']:
            results["is_valid"] = False
            results["errors"].append(
                f"Dataset has only {len(df)} rows. Minimum required: {self.validation_rules['min_rows']}"
            )
        
        # Check minimum columns
        if len(df.columns) < self.validation_rules['min_columns']:
            results["is_valid"] = False
            results["errors"].append(
                f"Dataset has only {len(df.columns)} columns. Minimum required: {self.validation_rules['min_columns']}"
            )
        
        # Check for empty dataset
        if df.empty:
            results["is_valid"] = False
            results["errors"].append("Dataset is empty")
        
        # Check for duplicate column names
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            results["warnings"].append(f"Duplicate column names found: {duplicate_cols}")
            results["recommendations"].append("Rename duplicate columns to ensure uniqueness")
        
        # Check column name validity
        invalid_col_names = []
        for col in df.columns:
            if not isinstance(col, str) or col.strip() == "":
                invalid_col_names.append(col)
        
        if invalid_col_names:
            results["warnings"].append(f"Invalid column names found: {invalid_col_names}")
            results["recommendations"].append("Ensure all column names are non-empty strings")
        
        return results
    
    async def _validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality aspects"""
        
        results = {
            "is_valid": True,
            "warnings": [],
            "recommendations": []
        }
        
        # Check missing values
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        
        if missing_percentage > self.validation_rules['max_missing_percentage']:
            results["warnings"].append(
                f"Dataset has {missing_percentage:.1f}% missing values (threshold: {self.validation_rules['max_missing_percentage']}%)"
            )
            results["recommendations"].append("Consider data collection improvements or imputation strategies")
        elif missing_percentage > 50:
            results["warnings"].append(f"High missing value percentage: {missing_percentage:.1f}%")
            results["recommendations"].append("Review missing value patterns and consider advanced imputation")
        
        # Check duplicate rows
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(df)) * 100
        
        if duplicate_percentage > self.validation_rules['max_duplicate_percentage']:
            results["warnings"].append(
                f"Dataset has {duplicate_percentage:.1f}% duplicate rows"
            )
            results["recommendations"].append("Remove duplicate rows before analysis")
        elif duplicate_count > 0:
            results["warnings"].append(f"Found {duplicate_count} duplicate rows ({duplicate_percentage:.1f}%)")
            results["recommendations"].append("Consider removing duplicate rows")
        
        # Check for constant columns
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() <= self.validation_rules['min_unique_values']:
                constant_cols.append(col)
        
        if constant_cols:
            results["warnings"].append(f"Constant/near-constant columns found: {constant_cols}")
            results["recommendations"].append("Consider removing constant columns as they provide no information")
        
        # Check data types
        mixed_type_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column has mixed types
                sample_types = set(type(x).__name__ for x in df[col].dropna().head(100))
                if len(sample_types) > 1:
                    mixed_type_cols.append(col)
        
        if mixed_type_cols:
            results["warnings"].append(f"Columns with mixed data types: {mixed_type_cols}")
            results["recommendations"].append("Standardize data types for consistent processing")
        
        return results
    
    async def _validate_ml_readiness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataset readiness for ML"""
        
        results = {
            "warnings": [],
            "recommendations": []
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Check feature diversity
        if len(numeric_cols) == 0:
            results["warnings"].append("No numeric columns found")
            results["recommendations"].append("Ensure dataset has numeric features for ML algorithms")
        
        # Check high cardinality categorical columns
        high_cardinality_cols = []
        for col in categorical_cols:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.8:
                high_cardinality_cols.append(col)
        
        if high_cardinality_cols:
            results["warnings"].append(f"High cardinality categorical columns: {high_cardinality_cols}")
            results["recommendations"].append("Consider feature engineering for high cardinality categorical variables")
        
        # Check for potential target leakage
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        if datetime_cols:
            results["recommendations"].append(
                f"Datetime columns found: {datetime_cols}. Ensure no future information leakage in target prediction"
            )
        
        # Check for skewed distributions in numeric columns
        highly_skewed_cols = []
        for col in numeric_cols:
            if abs(df[col].skew()) > 3:
                highly_skewed_cols.append(col)
        
        if highly_skewed_cols:
            results["warnings"].append(f"Highly skewed numeric columns: {highly_skewed_cols}")
            results["recommendations"].append("Consider log transformation or other normalization for skewed features")
        
        # Check for outliers
        outlier_cols = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_percentage = (outlier_count / len(df)) * 100
                
                if outlier_percentage > 10:
                    outlier_cols.append(col)
        
        if outlier_cols:
            results["warnings"].append(f"Columns with significant outliers (>10%): {outlier_cols}")
            results["recommendations"].append("Consider outlier treatment strategies")
        
        return results
    
    def validate_target_column(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Validate target column for ML"""
        
        results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "target_info": {}
        }
        
        if target_column not in df.columns:
            results["is_valid"] = False
            results["errors"].append(f"Target column '{target_column}' not found in dataset")
            return results
        
        target_series = df[target_column]
        
        # Basic target info
        results["target_info"] = {
            "column_name": target_column,
            "data_type": str(target_series.dtype),
            "unique_values": target_series.nunique(),
            "missing_values": target_series.isnull().sum(),
            "missing_percentage": (target_series.isnull().sum() / len(target_series)) * 100
        }
        
        # Check for missing values in target
        if target_series.isnull().sum() > 0:
            missing_pct = (target_series.isnull().sum() / len(target_series)) * 100
            if missing_pct > 5:
                results["warnings"].append(f"Target column has {missing_pct:.1f}% missing values")
                results["recommendations"].append("Handle missing values in target column before training")
        
        # Check target distribution
        if pd.api.types.is_numeric_dtype(target_series):
            # Numeric target (regression)
            if target_series.nunique() < 10:
                results["warnings"].append("Numeric target has very few unique values - consider classification")
            
            if abs(target_series.skew()) > 2:
                results["warnings"].append("Target variable is highly skewed")
                results["recommendations"].append("Consider log transformation of target variable")
        
        else:
            # Categorical target (classification)
            value_counts = target_series.value_counts()
            
            # Check class balance
            if len(value_counts) > 1:
                balance_ratio = value_counts.iloc[-1] / value_counts.iloc[0]
                if balance_ratio < 0.1:
                    results["warnings"].append(f"Severe class imbalance detected (ratio: {balance_ratio:.3f})")
                    results["recommendations"].append("Consider class balancing techniques (SMOTE, class weights)")
                elif balance_ratio < 0.3:
                    results["warnings"].append(f"Class imbalance detected (ratio: {balance_ratio:.3f})")
                    results["recommendations"].append("Monitor model performance across all classes")
            
            # Check for too many classes
            if len(value_counts) > 50:
                results["warnings"].append(f"Target has {len(value_counts)} classes - very high cardinality")
                results["recommendations"].append("Consider grouping rare classes or using hierarchical classification")
        
        return results
