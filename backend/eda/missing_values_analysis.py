"""Advanced missing values analysis and visualization."""
import pandas as pd
import numpy as np
import io
import base64
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add the project root to the path so we can import from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.ai_utils import generate_insight, AIProvider

def run(df, params=None):
    """
    Analyze missing values in the dataset with advanced visualizations and AI insights
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to analyze
    params : dict, optional
        Additional parameters (not used in this function)
        
    Returns:
    --------
    dict
        Results of the analysis including:
        - missing_stats: Basic missing value statistics
        - missing_patterns: Analysis of missing value patterns
        - visualizations: Base64 encoded visualizations
        - ai_insights: AI-generated insights about missing values
    """
    # Basic missing value statistics
    missing_stats = {
        "total_missing": df.isnull().sum().sum(),
        "missing_by_column": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
        "rows_with_missing": (df.isnull().sum(axis=1) > 0).sum(),
        "rows_with_missing_percentage": (df.isnull().sum(axis=1) > 0).sum() / len(df) * 100
    }
    
    # Missing value patterns
    missing_patterns = {
        "missing_pattern_counts": df.isnull().sum(axis=1).value_counts().to_dict(),
        "columns_with_missing": [col for col in df.columns if df[col].isnull().any()],
        "columns_without_missing": [col for col in df.columns if not df[col].isnull().any()]
    }
    
    # Create visualizations
    visualizations = {}
    
    # 1. Missing values heatmap
    if missing_stats["total_missing"] > 0:
        # Create a heatmap of missing values
        fig = go.Figure()
        
        # Create a binary mask of missing values (True where missing)
        mask = df.isnull()
        
        # Convert to 0 and 1 for visualization
        heatmap_data = mask.astype(int)
        
        # Add heatmap trace
        fig.add_trace(go.Heatmap(
            z=heatmap_data.values.T,
            x=np.arange(len(df)),
            y=df.columns,
            colorscale=[[0, 'white'], [1, 'red']],
            showscale=False,
            hovertemplate='Row: %{x}<br>Column: %{y}<br>Missing: %{z}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title='Missing Values Heatmap',
            xaxis_title='Row Index',
            yaxis_title='Column',
            height=max(400, len(df.columns) * 25),
            width=800,
            yaxis=dict(autorange='reversed')
        )
        
        # Convert to base64 for frontend display
        buffer = io.BytesIO()
        fig.write_image(buffer, format="png")
        buffer.seek(0)
        visualizations["missing_heatmap"] = base64.b64encode(buffer.read()).decode('utf-8')
    
    # 2. Missing values bar chart
    if missing_stats["total_missing"] > 0:
        # Create a bar chart of missing values by column
        missing_df = pd.DataFrame({
            'Column': list(missing_stats["missing_percentage"].keys()),
            'Missing (%)': list(missing_stats["missing_percentage"].values())
        })
        
        # Sort by missing percentage
        missing_df = missing_df.sort_values('Missing (%)', ascending=False)
        
        # Only include columns with missing values
        missing_df = missing_df[missing_df['Missing (%)'] > 0]
        
        if not missing_df.empty:
            fig = px.bar(
                missing_df,
                x='Column',
                y='Missing (%)',
                title='Missing Values by Column',
                color='Missing (%)',
                color_continuous_scale='Reds',
                text='Missing (%)',
                height=max(400, len(missing_df) * 25),
                width=800
            )
            
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            
            # Convert to base64 for frontend display
            buffer = io.BytesIO()
            fig.write_image(buffer, format="png")
            buffer.seek(0)
            visualizations["missing_barchart"] = base64.b64encode(buffer.read()).decode('utf-8')
    
    # 3. Missing values correlation matrix
    if len(missing_patterns["columns_with_missing"]) > 1:
        # Create a correlation matrix of missing value patterns
        missing_mask = df[missing_patterns["columns_with_missing"]].isnull()
        corr_matrix = missing_mask.corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            title='Missing Value Correlation Matrix',
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            height=max(400, len(corr_matrix) * 30),
            width=max(400, len(corr_matrix) * 30)
        )
        
        # Convert to base64 for frontend display
        buffer = io.BytesIO()
        fig.write_image(buffer, format="png")
        buffer.seek(0)
        visualizations["missing_correlation"] = base64.b64encode(buffer.read()).decode('utf-8')
    
    # Generate AI insights about missing values
    # Analyze missing data patterns (MCAR, MAR, or MNAR)
    missing_pattern_analysis = analyze_missing_patterns(df)
    
    # Generate AI insights
    insight_prompt = f"""
    Analyze the missing value patterns in this dataset:
    - Total missing values: {missing_stats['total_missing']}
    - Rows with missing values: {missing_stats['rows_with_missing']} ({missing_stats['rows_with_missing_percentage']:.1f}%)
    - Missing percentages by column: {missing_stats['missing_percentage']}
    - Missing value patterns: {missing_patterns['missing_pattern_counts']}
    - Missing value correlation analysis: {missing_pattern_analysis['pattern_type']}
    
    Please provide:
    1. Assessment of missing data patterns (MCAR, MAR, or MNAR)
    2. Recommendations for handling missing values based on the pattern
    3. Potential impact on analysis if not addressed properly
    4. Specific recommendations for each column with significant missing data
    """
    
    try:
        ai_insights = generate_insight(insight_prompt, context=missing_stats, provider=AIProvider.GEMINI)
    except Exception as e:
        ai_insights = f"Unable to generate AI insights: {str(e)}"
    
    # Return results
    return {
        'missing_stats': missing_stats,
        'missing_patterns': missing_patterns,
        'missing_pattern_analysis': missing_pattern_analysis,
        'visualizations': visualizations,
        'ai_insights': ai_insights
    }

def analyze_missing_patterns(df):
    """
    Analyze missing data patterns to determine if they are MCAR, MAR, or MNAR
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to analyze
        
    Returns:
    --------
    dict
        Results of the analysis including pattern type and evidence
    """
    # Get columns with missing values
    cols_with_missing = [col for col in df.columns if df[col].isnull().any()]
    
    if not cols_with_missing:
        return {
            'pattern_type': 'No missing values',
            'evidence': 'The dataset has no missing values.'
        }
    
    # Create binary indicators for missing values
    missing_indicators = pd.DataFrame()
    for col in cols_with_missing:
        missing_indicators[f'{col}_missing'] = df[col].isnull().astype(int)
    
    # Check for correlations between missing indicators
    corr_matrix = missing_indicators.corr()
    
    # Check for strong correlations (absolute value > 0.5)
    strong_correlations = (corr_matrix.abs() > 0.5).sum().sum() - len(cols_with_missing)  # Subtract diagonal
    
    # Check for correlations between missing indicators and non-missing values
    # For each column with missing values, check if missingness correlates with other columns' values
    mar_evidence = []
    
    for col in cols_with_missing:
        # Get numeric columns that don't have missing values
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != col and c not in cols_with_missing]
        
        if numeric_cols:
            # Create a temporary DataFrame with the missing indicator and numeric columns
            temp_df = pd.DataFrame()
            temp_df['missing'] = df[col].isnull().astype(int)
            
            for num_col in numeric_cols[:5]:  # Limit to 5 columns for efficiency
                temp_df[num_col] = df[num_col]
            
            # Calculate point-biserial correlation (equivalent to Pearson for binary and continuous)
            correlations = temp_df.corr()['missing'].drop('missing').abs()
            
            # Check if any correlations are strong (> 0.3)
            strong_cols = correlations[correlations > 0.3].index.tolist()
            if strong_cols:
                mar_evidence.append(f"Missingness in '{col}' correlates with values in {', '.join(strong_cols)}")
    
    # Determine pattern type based on evidence
    if strong_correlations > 0 or mar_evidence:
        pattern_type = 'MAR (Missing At Random)'
        evidence = 'Evidence suggests missingness depends on observed data:\n'
        if strong_correlations > 0:
            evidence += f"- Strong correlations between missing value patterns ({strong_correlations} pairs)\n"
        if mar_evidence:
            evidence += "- " + "\n- ".join(mar_evidence)
    else:
        # Perform Little's MCAR test (simplified version)
        # If no clear patterns are found, we assume MCAR, but note this is a simplification
        pattern_type = 'MCAR (Missing Completely At Random)'
        evidence = 'No strong evidence against MCAR was found. Missingness does not appear to depend on other observed variables.'
    
    # Note: MNAR cannot be definitively determined from the data alone
    # We can only suggest it might be MNAR if domain knowledge suggests it
    mnar_note = "\n\nNote: MNAR (Missing Not At Random) cannot be definitively determined from the data alone. " \
               "It requires domain knowledge about the data collection process."
    
    return {
        'pattern_type': pattern_type,
        'evidence': evidence + mnar_note
    }