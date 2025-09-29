"""
Comprehensive EDA Module

This module provides a unified interface for running comprehensive exploratory data analysis
on a pandas DataFrame, leveraging existing analysis modules.
"""
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import existing EDA modules
from eda.univariate_analysis import run as run_univariate_analysis
from eda.bivariate_analysis import run as run_bivariate_analysis
from eda.missing_values_analysis import analyze_missing_values
from eda.outlier_detection_enhanced import detect_outliers
from eda.correlation import calculate_correlations
from eda.data_distribution import analyze_distributions

class ComprehensiveEDA:
    """
    A comprehensive EDA class that combines various analysis modules
    into a single, easy-to-use interface.
    """
    
    def __init__(self, df: pd.DataFrame, output_dir: str = "eda_output"):
        """
        Initialize the ComprehensiveEDA with a DataFrame.
        
        Args:
            df: Input pandas DataFrame
            output_dir: Directory to save output files and visualizations
        """
        self.df = df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
        # Initialize analysis results
        self.analysis_results = {
            "dataset_info": {},
            "univariate_analysis": {},
            "bivariate_analysis": {},
            "missing_values": {},
            "outliers": {},
            "correlations": {},
            "distributions": {}
        }
    
    def run_all_analyses(self) -> Dict[str, Any]:
        """Run all available EDA analyses."""
        try:
            # 1. Basic dataset information
            self.analysis_results["dataset_info"] = self._get_dataset_info()
            
            # 2. Univariate analysis
            print("Running univariate analysis...")
            self.analysis_results["univariate_analysis"] = run_univariate_analysis(self.df)
            
            # 3. Bivariate analysis
            print("Running bivariate analysis...")
            self.analysis_results["bivariate_analysis"] = run_bivariate_analysis(self.df)
            
            # 4. Missing values analysis
            print("Analyzing missing values...")
            self.analysis_results["missing_values"] = analyze_missing_values(self.df)
            
            # 5. Outlier detection
            print("Detecting outliers...")
            self.analysis_results["outliers"] = detect_outliers(self.df)
            
            # 6. Correlation analysis
            print("Calculating correlations...")
            self.analysis_results["correlations"] = calculate_correlations(self.df)
            
            # 7. Distribution analysis
            print("Analyzing distributions...")
            self.analysis_results["distributions"] = analyze_distributions(self.df)
            
            # Save results
            self._save_results()
            
            return self.analysis_results
            
        except Exception as e:
            return {"error": f"Error during EDA: {str(e)}"}
    
    def _get_dataset_info(self) -> Dict[str, Any]:
        """Get basic information about the dataset."""
        return {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "data_types": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "memory_usage": f"{self.df.memory_usage(deep=True).sum() / (1024**2):.2f} MB",
            "duplicate_rows": self.df.duplicated().sum(),
            "total_missing_values": self.df.isnull().sum().sum(),
            "missing_values_percentage": f"{(self.df.isnull().sum().sum() / np.product(self.df.shape)) * 100:.2f}%"
        }
    
    def _save_results(self) -> None:
        """Save analysis results to files."""
        # Save full results as JSON
        with open(self.output_dir / "eda_results.json", "w") as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        # Save summary report
        self._generate_summary_report()
    
    def _generate_summary_report(self) -> None:
        """Generate a summary report of the EDA."""
        summary = {
            "dataset_summary": self.analysis_results["dataset_info"],
            "key_insights": {
                "columns_with_missing_values": [
                    col for col in self.analysis_results["missing_values"]["missing_counts"] 
                    if self.analysis_results["missing_values"]["missing_counts"][col] > 0
                ],
                "columns_with_outliers": list(self.analysis_results["outliers"].get("outlier_columns", [])),
                "highly_correlated_pairs": [
                    f"{pair[0]} & {pair[1]} ({corr:.2f})" 
                    for pair, corr in self.analysis_results["correlations"].get("highly_correlated_pairs", [])
                ]
            },
            "recommendations": self._generate_recommendations()
        }
        
        with open(self.output_dir / "summary_report.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate data cleaning and preprocessing recommendations."""
        recommendations = []
        
        # Missing values recommendations
        missing_info = self.analysis_results["missing_values"]
        for col, count in missing_info.get("missing_counts", {}).items():
            if count > 0:
                recommendations.append(
                    f"Consider handling missing values in column '{col}' "
                    f"({count} missing values, {missing_info['missing_percentages'][col]:.1f}% of data)"
                )
        
        # Outlier recommendations
        outliers = self.analysis_results.get("outliers", {})
        for col in outliers.get("outlier_columns", []):
            recommendations.append(
                f"Investigate potential outliers in column '{col}'. "
                f"Detected {outliers['outlier_counts'].get(col, 0)} potential outliers."
            )
        
        # Data type recommendations
        for col, dtype in self.analysis_results["dataset_info"]["data_types"].items():
            if "object" in dtype and self.df[col].nunique() < 20:
                recommendations.append(
                    f"Consider converting column '{col}' to categorical data type "
                    "as it has a limited number of unique values."
                )
        
        return recommendations


def run_comprehensive_eda(df: pd.DataFrame, output_dir: str = "eda_output") -> Dict[str, Any]:
    """
    Run comprehensive EDA on a pandas DataFrame.
    
    Args:
        df: Input pandas DataFrame
        output_dir: Directory to save output files and visualizations
        
    Returns:
        Dictionary containing all EDA results
    """
    eda = ComprehensiveEDA(df, output_dir)
    return eda.run_all_analyses()


if __name__ == "__main__":
    # Example usage
    import seaborn as sns
    
    print("Loading example dataset (Titanic)...")
    df = sns.load_dataset("titanic")
    
    print("Running comprehensive EDA...")
    results = run_comprehensive_eda(df)
    
    print(f"\nEDA completed! Results saved to 'eda_output' directory.")
    print("\nDataset Summary:")
    print(f"- Rows: {results['dataset_info']['shape'][0]}")
    print(f"- Columns: {results['dataset_info']['shape'][1]}")
    print(f"- Total missing values: {results['dataset_info']['total_missing_values']}")
    print("\nKey insights and recommendations have been saved to 'summary_report.json'")
