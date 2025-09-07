import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import missingno as msno
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from ydata_profiling import ProfileReport
    YDATA_AVAILABLE = True
except ImportError:
    YDATA_AVAILABLE = False

try:
    import sweetviz as sv
    SWEETVIZ_AVAILABLE = True
except ImportError:
    SWEETVIZ_AVAILABLE = False

class ProfilingAgent:
    """Agent for generating comprehensive data profiling reports"""
    
    def __init__(self):
        self.name = "Profiling Agent"
        
    def process(self, context) -> Dict:
        """
        Generate profiling reports and visualizations
        
        Returns:
            Dict with report paths and inline visualizations
        """
        df = context.df
        results = {
            'html_reports': {},
            'visualizations': {},
            'summary_stats': {}
        }
        
        # Create timestamped report directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path("reports") / timestamp
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate HTML reports
        results['html_reports'] = self._generate_html_reports(df, report_dir, context.filename)
        
        # Generate inline visualizations
        results['visualizations'] = self._generate_visualizations(df)
        
        # Summary statistics
        results['summary_stats'] = self._generate_summary_stats(df)
        
        return results
    
    def _generate_html_reports(self, df: pd.DataFrame, report_dir: Path, filename: str) -> Dict:
        """Generate ydata-profiling and sweetviz HTML reports"""
        html_reports = {}
        
        # ydata-profiling report
        if YDATA_AVAILABLE:
            try:
                profile = ProfileReport(
                    df, 
                    title=f"Data Profile: {filename}",
                    minimal=True,
                    explorative=True
                )
                ydata_path = report_dir / "ydata_profile.html"
                profile.to_file(ydata_path)
                html_reports['ydata'] = str(ydata_path)
            except Exception as e:
                html_reports['ydata_error'] = str(e)
        
        # Sweetviz report
        if SWEETVIZ_AVAILABLE:
            try:
                report = sv.analyze(df)
                sweetviz_path = report_dir / "sweetviz_report.html"
                report.show_html(str(sweetviz_path), open_browser=False)
                html_reports['sweetviz'] = str(sweetviz_path)
            except Exception as e:
                html_reports['sweetviz_error'] = str(e)
                
        return html_reports
    
    def _generate_visualizations(self, df: pd.DataFrame) -> Dict:
        """Generate inline visualizations for Streamlit"""
        viz = {}
        
        # Missing data visualization
        viz['missing'] = self._plot_missing_data(df)
        
        # Distribution plots for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            viz['distributions'] = self._plot_distributions(df[numeric_cols])
        
        # Categorical distributions
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            viz['categorical'] = self._plot_categorical(df[categorical_cols])
            
        return viz
    
    def _plot_missing_data(self, df: pd.DataFrame):
        """Create missing data visualizations"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Missing Data Matrix', 'Missing Data by Column'),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Missing data heatmap (simplified)
        missing_data = df.isnull()
        missing_counts = missing_data.sum()
        missing_pcts = (missing_counts / len(df)) * 100
        
        # Bar chart of missing percentages
        fig.add_trace(
            go.Bar(
                x=missing_pcts.index,
                y=missing_pcts.values,
                name='Missing %',
                marker_color='red'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=800,
            title_text="Missing Data Analysis",
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Columns", row=2, col=1)
        fig.update_yaxes(title_text="Missing %", row=2, col=1)
        
        return fig
    
    def _plot_distributions(self, df_numeric: pd.DataFrame):
        """Create distribution plots for numeric columns"""
        n_cols = min(4, len(df_numeric.columns))
        n_rows = (len(df_numeric.columns) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=df_numeric.columns[:n_rows*n_cols]
        )
        
        for i, col in enumerate(df_numeric.columns[:n_rows*n_cols]):
            row = i // n_cols + 1
            col_idx = i % n_cols + 1
            
            fig.add_trace(
                go.Histogram(
                    x=df_numeric[col].dropna(),
                    name=col,
                    showlegend=False
                ),
                row=row, col=col_idx
            )
        
        fig.update_layout(
            height=300 * n_rows,
            title_text="Numeric Distributions"
        )
        
        return fig
    
    def _plot_categorical(self, df_cat: pd.DataFrame):
        """Create categorical distribution plots"""
        # Show top categories for categorical columns with reasonable cardinality
        figs = {}
        
        for col in df_cat.columns:
            if df_cat[col].nunique() <= 20:  # Only plot if <= 20 unique values
                value_counts = df_cat[col].value_counts().head(10)
                
                fig = go.Figure(data=[
                    go.Bar(x=value_counts.values, y=value_counts.index, orientation='h')
                ])
                
                fig.update_layout(
                    title=f"Top Values: {col}",
                    xaxis_title="Count",
                    yaxis_title="Categories",
                    height=400
                )
                
                figs[col] = fig
                
        return figs
    
    def _generate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics"""
        stats = {
            'shape': df.shape,
            'missing_summary': {
                'total_missing': df.isnull().sum().sum(),
                'columns_with_missing': (df.isnull().sum() > 0).sum(),
                'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            },
            'dtypes_summary': df.dtypes.value_counts().to_dict(),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        }
        
        # Numeric summary
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            stats['numeric_summary'] = numeric_df.describe().to_dict()
            
        return stats


            
