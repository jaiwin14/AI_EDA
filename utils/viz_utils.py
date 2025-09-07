import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings

warnings.filterwarnings('ignore')

# Set style defaults
plt.style.use('default')
sns.set_palette("husl")

def create_correlation_heatmap(corr_matrix: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    """
    Create an interactive correlation heatmap using Plotly
    
    Args:
        corr_matrix: Correlation matrix DataFrame
        title: Plot title
        
    Returns:
        Plotly figure
    """
    try:
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Features",
            yaxis_title="Features",
            width=600,
            height=500
        )
        
        return fig
        
    except Exception as e:
        # Return empty figure on error
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating heatmap: {str(e)}", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

def create_missing_data_visualization(df: pd.DataFrame) -> go.Figure:
    """
    Create missing data visualization
    
    Args:
        df: Input DataFrame
        
    Returns:
        Plotly figure showing missing data patterns
    """
    try:
        # Calculate missing data percentage
        missing_data = df.isnull().sum().sort_values(ascending=True)
        missing_pct = (missing_data / len(df) * 100)
        
        # Filter out columns with no missing data
        missing_pct = missing_pct[missing_pct > 0]
        
        if len(missing_pct) == 0:
            # No missing data
            fig = go.Figure()
            fig.add_annotation(text="No Missing Data Found!", 
                              xref="paper", yref="paper", x=0.5, y=0.5,
                              font=dict(size=20, color="green"))
            fig.update_layout(title="Missing Data Analysis")
            return fig
        
        # Create horizontal bar chart
        fig = go.Figure(go.Bar(
            y=missing_pct.index,
            x=missing_pct.values,
            orientation='h',
            marker_color='salmon',
            text=[f"{val:.1f}%" for val in missing_pct.values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Missing Data by Column",
            xaxis_title="Missing Percentage (%)",
            yaxis_title="Columns",
            height=max(400, len(missing_pct) * 30)
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating missing data viz: {str(e)}", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

def create_distribution_plots(df: pd.DataFrame, max_cols: int = 6) -> go.Figure:
    """
    Create distribution plots for numeric columns
    
    Args:
        df: Input DataFrame
        max_cols: Maximum number of columns to plot
        
    Returns:
        Plotly figure with subplots
    """
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:max_cols]
        
        if len(numeric_cols) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No Numeric Columns Found", 
                              xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        # Calculate subplot grid
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=numeric_cols,
            vertical_spacing=0.1
        )
        
        for i, col in enumerate(numeric_cols):
            row = i // n_cols + 1
            col_num = i % n_cols + 1
            
            # Create histogram
            data = df[col].dropna()
            fig.add_trace(
                go.Histogram(x=data, name=col, showlegend=False),
                row=row, col=col_num
            )
        
        fig.update_layout(
            title="Distribution of Numeric Variables",
            height=300 * n_rows
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating distributions: {str(e)}", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

def create_categorical_plots(df: pd.DataFrame, max_cols: int = 4) -> Dict[str, go.Figure]:
    """
    Create bar plots for categorical columns
    
    Args:
        df: Input DataFrame
        max_cols: Maximum number of columns to plot
        
    Returns:
        Dictionary of column names to Plotly figures
    """
    try:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns[:max_cols]
        plots = {}
        
        for col in cat_cols:
            # Get value counts
            value_counts = df[col].value_counts().head(20)  # Top 20 categories
            
            fig = go.Figure(go.Bar(
                x=value_counts.values,
                y=value_counts.index,
                orientation='h',
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title=f"Distribution of {col}",
                xaxis_title="Count",
                yaxis_title="Categories",
                height=max(400, len(value_counts) * 25)
            )
            
            plots[col] = fig
        
        return plots
        
    except Exception as e:
        return {'error': go.Figure().add_annotation(
            text=f"Error creating categorical plots: {str(e)}", 
            xref="paper", yref="paper", x=0.5, y=0.5
        )}

def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, 
                       color_col: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
    """
    Create scatter plot
    
    Args:
        df: Input DataFrame
        x_col: X-axis column
        y_col: Y-axis column
        color_col: Optional column for coloring points
        title: Plot title
        
    Returns:
        Plotly figure
    """
    try:
        if title is None:
            title = f"{y_col} vs {x_col}"
        
        if color_col:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title)
        else:
            fig = px.scatter(df, x=x_col, y=y_col, title=title)
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating scatter plot: {str(e)}", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

def create_box_plots(df: pd.DataFrame, max_cols: int = 6) -> go.Figure:
    """
    Create box plots for numeric columns
    
    Args:
        df: Input DataFrame
        max_cols: Maximum number of columns to plot
        
    Returns:
        Plotly figure with box plots
    """
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:max_cols]
        
        if len(numeric_cols) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No Numeric Columns Found", 
                              xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        fig = go.Figure()
        
        for col in numeric_cols:
            fig.add_trace(go.Box(
                y=df[col].dropna(),
                name=col,
                boxpoints='outliers'
            ))
        
        fig.update_layout(
            title="Box Plots of Numeric Variables",
            yaxis_title="Values",
            height=500
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating box plots: {str(e)}", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

def create_feature_importance_plot(importance_df: pd.DataFrame, top_n: int = 15, 
                                 title: str = "Feature Importance") -> go.Figure:
    """
    Create feature importance plot
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
        title: Plot title
        
    Returns:
        Plotly figure
    """
    try:
        # Take top N features
        plot_data = importance_df.head(top_n).copy()
        
        # Sort by importance for plotting
        plot_data = plot_data.sort_values('importance', ascending=True)
        
        fig = go.Figure(go.Bar(
            y=plot_data['feature'],
            x=plot_data['importance'],
            orientation='h',
            marker_color='steelblue'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=max(400, len(plot_data) * 25)
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating importance plot: {str(e)}", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

def create_model_comparison_chart(models_performance: Dict[str, float], 
                                metric_name: str = "Score",
                                title: str = "Model Performance Comparison") -> go.Figure:
    """
    Create model comparison bar chart
    
    Args:
        models_performance: Dictionary of model names to scores
        metric_name: Name of the metric being compared
        title: Plot title
        
    Returns:
        Plotly figure
    """
    try:
        models = list(models_performance.keys())
        scores = list(models_performance.values())
        
        fig = go.Figure(go.Bar(
            x=models,
            y=scores,
            marker_color='lightcoral',
            text=[f"{score:.3f}" for score in scores],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Models",
            yaxis_title=metric_name,
            xaxis_tickangle=-45
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating comparison chart: {str(e)}", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

def save_figure(fig: go.Figure, filename: str, format: str = 'html') -> bool:
    """
    Save Plotly figure to file
    
    Args:
        fig: Plotly figure
        filename: Output filename
        format: Output format ('html', 'png', 'svg', etc.)
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        if format == 'html':
            fig.write_html(filename)
        elif format == 'png':
            fig.write_image(filename, format='png')
        elif format == 'svg':
            fig.write_image(filename, format='svg')
        else:
            return False
            
        return True
        
    except Exception as e:
        print(f"Error saving figure: {e}")
        return False

# Matplotlib helper functions for backward compatibility
def create_matplotlib_correlation_heatmap(corr_matrix: pd.DataFrame, 
                                        figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """Create correlation heatmap using matplotlib/seaborn"""
    try:
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation Matrix')
        return fig
    except Exception as e:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
        return fig