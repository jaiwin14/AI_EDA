"""
Visualization Generator
Create interactive Plotly visualizations for EDA and model results
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, Any, List, Optional
import logging
import json

logger = logging.getLogger(__name__)

class VisualizationGenerator:
    """Generate interactive Plotly visualizations for data analysis"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        self.template = "plotly_white"
    
    async def create_correlation_heatmap(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create correlation heatmap for numeric columns"""
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {"error": "No numeric columns found for correlation analysis"}
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            title="Feature Correlation Heatmap",
            color_continuous_scale="RdBu",
            aspect="auto",
            template=self.template
        )
        
        fig.update_layout(
            title_x=0.5,
            width=800,
            height=600,
            xaxis_title="Features",
            yaxis_title="Features"
        )
        
        # Add correlation values as text
        fig.update_traces(
            text=np.around(corr_matrix.values, decimals=2),
            texttemplate="%{text}",
            textfont={"size": 10}
        )
        
        return {
            "type": "correlation_heatmap",
            "figure": fig.to_dict(),
            "insights": self._analyze_correlation_insights(corr_matrix)
        }
    
    async def create_distribution_plots(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create distribution plots for numeric columns"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return {"error": "No numeric columns found for distribution analysis"}
        
        # Limit to first 12 columns for performance
        numeric_cols = numeric_cols[:12]
        
        # Calculate subplot dimensions
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        # Create subplots
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=numeric_cols,
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        for i, col in enumerate(numeric_cols):
            row = i // n_cols + 1
            col_idx = i % n_cols + 1
            
            # Create histogram
            fig.add_trace(
                go.Histogram(
                    x=df[col].dropna(),
                    name=col,
                    showlegend=False,
                    nbinsx=30,
                    opacity=0.7
                ),
                row=row,
                col=col_idx
            )
        
        fig.update_layout(
            title_text="Distribution of Numeric Variables",
            title_x=0.5,
            height=300 * n_rows,
            template=self.template
        )
        
        return {
            "type": "distribution_plots",
            "figure": fig.to_dict(),
            "insights": self._analyze_distribution_insights(df, numeric_cols)
        }
    
    async def create_missing_values_plot(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create missing values visualization"""
        
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        # Filter columns with missing values
        missing_data = missing_counts[missing_counts > 0]
        
        if missing_data.empty:
            return {
                "type": "missing_values",
                "message": "No missing values found in the dataset",
                "figure": None
            }
        
        # Create bar plot
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=missing_data.index,
            y=missing_data.values,
            name="Missing Count",
            marker_color=px.colors.sequential.Reds[4],
            text=missing_data.values,
            textposition='outside'
        ))
        
        # Add percentage on secondary y-axis
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=missing_data.index,
            y=missing_percentages[missing_data.index],
            mode='lines+markers',
            name="Missing %",
            yaxis='y2',
            line=dict(color=px.colors.sequential.Blues[6], width=3),
            marker=dict(size=8)
        ))
        
        # Combine both plots
        fig.add_trace(fig2.data[0])
        
        fig.update_layout(
            title="Missing Values Analysis",
            title_x=0.5,
            xaxis_title="Columns",
            yaxis_title="Missing Count",
            yaxis2=dict(
                title="Missing Percentage (%)",
                overlaying='y',
                side='right'
            ),
            template=self.template,
            height=500
        )
        
        fig.update_xaxis(tickangle=45)
        
        return {
            "type": "missing_values",
            "figure": fig.to_dict(),
            "insights": self._analyze_missing_insights(missing_counts, missing_percentages)
        }
    
    async def create_outlier_plots(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create outlier detection plots (box plots)"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return {"error": "No numeric columns found for outlier analysis"}
        
        # Limit to first 8 columns for performance
        numeric_cols = numeric_cols[:8]
        
        # Calculate subplot dimensions
        n_cols = min(2, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=numeric_cols,
            vertical_spacing=0.1
        )
        
        for i, col in enumerate(numeric_cols):
            row = i // n_cols + 1
            col_idx = i % n_cols + 1
            
            fig.add_trace(
                go.Box(
                    y=df[col].dropna(),
                    name=col,
                    showlegend=False,
                    boxpoints='outliers',
                    marker_color=self.color_palette[i % len(self.color_palette)]
                ),
                row=row,
                col=col_idx
            )
        
        fig.update_layout(
            title_text="Outlier Detection (Box Plots)",
            title_x=0.5,
            height=400 * n_rows,
            template=self.template
        )
        
        return {
            "type": "outlier_plots",
            "figure": fig.to_dict(),
            "insights": self._analyze_outlier_insights(df, numeric_cols)
        }
    
    async def create_pairplot(self, df: pd.DataFrame, max_features: int = 6) -> Dict[str, Any]:
        """Create pairplot for numeric features"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return {"error": "Need at least 2 numeric columns for pairplot"}
        
        # Limit features for performance
        selected_cols = numeric_cols[:max_features]
        subset_df = df[selected_cols].dropna()
        
        if len(subset_df) == 0:
            return {"error": "No data available after removing missing values"}
        
        # Create scatter plot matrix
        fig = px.scatter_matrix(
            subset_df,
            dimensions=selected_cols,
            title="Feature Pairplot",
            template=self.template
        )
        
        fig.update_layout(
            title_x=0.5,
            height=800,
            width=800
        )
        
        return {
            "type": "pairplot",
            "figure": fig.to_dict(),
            "features_included": selected_cols
        }
    
    async def create_feature_importance_plot(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create feature importance visualization based on variance"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return {"error": "No numeric columns found for feature importance analysis"}
        
        # Calculate variance-based importance
        variances = df[numeric_cols].var().sort_values(ascending=True)
        
        # Normalize to 0-1 scale
        normalized_importance = (variances - variances.min()) / (variances.max() - variances.min())
        
        fig = go.Figure(go.Bar(
            x=normalized_importance.values,
            y=normalized_importance.index,
            orientation='h',
            marker_color=px.colors.sequential.Viridis,
            text=np.round(normalized_importance.values, 3),
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Feature Importance (Variance-based)",
            title_x=0.5,
            xaxis_title="Normalized Importance",
            yaxis_title="Features",
            template=self.template,
            height=max(400, len(numeric_cols) * 30)
        )
        
        return {
            "type": "feature_importance",
            "figure": fig.to_dict(),
            "method": "variance_based",
            "importance_scores": normalized_importance.to_dict()
        }
    
    async def create_target_analysis_plot(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Create target variable analysis plots"""
        
        if target_col not in df.columns:
            return {"error": f"Target column '{target_col}' not found in dataset"}
        
        target_series = df[target_col].dropna()
        
        if len(target_series) == 0:
            return {"error": "Target column has no valid values"}
        
        # Determine if target is categorical or numeric
        is_categorical = pd.api.types.is_object_dtype(target_series) or target_series.nunique() < 20
        
        if is_categorical:
            # Create bar plot for categorical target
            value_counts = target_series.value_counts()
            
            fig = go.Figure(go.Bar(
                x=value_counts.index.astype(str),
                y=value_counts.values,
                marker_color=px.colors.sequential.Blues,
                text=value_counts.values,
                textposition='outside'
            ))
            
            fig.update_layout(
                title=f"Target Distribution: {target_col}",
                title_x=0.5,
                xaxis_title=target_col,
                yaxis_title="Count",
                template=self.template
            )
            
            analysis_type = "categorical"
            
        else:
            # Create histogram for numeric target
            fig = go.Figure(go.Histogram(
                x=target_series,
                nbinsx=30,
                marker_color=px.colors.sequential.Blues[4],
                opacity=0.7
            ))
            
            # Add statistical lines
            mean_val = target_series.mean()
            median_val = target_series.median()
            
            fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                         annotation_text=f"Mean: {mean_val:.2f}")
            fig.add_vline(x=median_val, line_dash="dot", line_color="green",
                         annotation_text=f"Median: {median_val:.2f}")
            
            fig.update_layout(
                title=f"Target Distribution: {target_col}",
                title_x=0.5,
                xaxis_title=target_col,
                yaxis_title="Frequency",
                template=self.template
            )
            
            analysis_type = "numeric"
        
        return {
            "type": "target_analysis",
            "figure": fig.to_dict(),
            "target_column": target_col,
            "analysis_type": analysis_type,
            "insights": self._analyze_target_insights(target_series, analysis_type)
        }
    
    def _analyze_correlation_insights(self, corr_matrix: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract insights from correlation matrix"""
        
        insights = []
        
        # Find strong correlations
        strong_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= 0.7:
                    strong_pairs.append({
                        "feature_1": corr_matrix.columns[i],
                        "feature_2": corr_matrix.columns[j],
                        "correlation": round(corr_val, 3)
                    })
        
        if strong_pairs:
            insights.append({
                "type": "strong_correlations",
                "message": f"Found {len(strong_pairs)} strong correlations (|r| ≥ 0.7)",
                "details": strong_pairs[:5]  # Top 5
            })
        
        # Check for multicollinearity
        high_corr_count = sum(1 for i in range(len(corr_matrix.columns)) 
                             for j in range(i+1, len(corr_matrix.columns))
                             if abs(corr_matrix.iloc[i, j]) >= 0.9)
        
        if high_corr_count > 0:
            insights.append({
                "type": "multicollinearity_warning",
                "message": f"Potential multicollinearity detected in {high_corr_count} feature pairs",
                "recommendation": "Consider feature selection or dimensionality reduction"
            })
        
        return insights
    
    def _analyze_distribution_insights(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[Dict[str, Any]]:
        """Extract insights from distribution analysis"""
        
        insights = []
        
        # Check for skewed distributions
        skewed_cols = []
        for col in numeric_cols:
            skewness = df[col].skew()
            if abs(skewness) > 1:
                skewed_cols.append({"column": col, "skewness": round(skewness, 3)})
        
        if skewed_cols:
            insights.append({
                "type": "skewed_distributions",
                "message": f"{len(skewed_cols)} columns have skewed distributions",
                "details": skewed_cols,
                "recommendation": "Consider log transformation or other normalization techniques"
            })
        
        return insights
    
    def _analyze_missing_insights(self, missing_counts: pd.Series, missing_percentages: pd.Series) -> List[Dict[str, Any]]:
        """Extract insights from missing values analysis"""
        
        insights = []
        
        # High missing value columns
        high_missing = missing_percentages[missing_percentages > 50]
        if not high_missing.empty:
            insights.append({
                "type": "high_missing_values",
                "message": f"{len(high_missing)} columns have >50% missing values",
                "columns": high_missing.index.tolist(),
                "recommendation": "Consider dropping these columns or investigating data collection issues"
            })
        
        # Medium missing value columns
        medium_missing = missing_percentages[(missing_percentages > 20) & (missing_percentages <= 50)]
        if not medium_missing.empty:
            insights.append({
                "type": "medium_missing_values",
                "message": f"{len(medium_missing)} columns have 20-50% missing values",
                "columns": medium_missing.index.tolist(),
                "recommendation": "Consider advanced imputation techniques"
            })
        
        return insights
    
    def _analyze_outlier_insights(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[Dict[str, Any]]:
        """Extract insights from outlier analysis"""
        
        insights = []
        
        # Count outliers using IQR method
        outlier_cols = []
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue
                
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            outlier_percentage = (len(outliers) / len(series)) * 100
            
            if outlier_percentage > 5:
                outlier_cols.append({
                    "column": col,
                    "outlier_percentage": round(outlier_percentage, 2),
                    "outlier_count": len(outliers)
                })
        
        if outlier_cols:
            insights.append({
                "type": "outlier_detection",
                "message": f"{len(outlier_cols)} columns have significant outliers (>5%)",
                "details": outlier_cols,
                "recommendation": "Consider outlier treatment or robust scaling methods"
            })
        
        return insights
    
    def _analyze_target_insights(self, target_series: pd.Series, analysis_type: str) -> List[Dict[str, Any]]:
        """Extract insights from target variable analysis"""
        
        insights = []
        
        if analysis_type == "categorical":
            value_counts = target_series.value_counts()
            
            # Check class imbalance
            balance_ratio = value_counts.iloc[-1] / value_counts.iloc[0] if len(value_counts) > 1 else 1.0
            
            if balance_ratio < 0.3:
                insights.append({
                    "type": "class_imbalance",
                    "message": f"Significant class imbalance detected (ratio: {balance_ratio:.3f})",
                    "recommendation": "Consider using stratified sampling, SMOTE, or class weights"
                })
            
            if len(value_counts) > 10:
                insights.append({
                    "type": "high_cardinality",
                    "message": f"High number of classes ({len(value_counts)})",
                    "recommendation": "Consider grouping rare classes or hierarchical classification"
                })
        
        else:  # numeric
            skewness = target_series.skew()
            
            if abs(skewness) > 1:
                insights.append({
                    "type": "skewed_target",
                    "message": f"Target variable is skewed (skewness: {skewness:.3f})",
                    "recommendation": "Consider log transformation or other normalization techniques"
                })
        
        return insights
