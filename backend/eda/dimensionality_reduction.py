"""Dimensionality Reduction Analysis Module with AI Recommendations"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.ai_utils import generate_insight
from utils.serialization_fixed import to_json_serializable

def run(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the need for dimensionality reduction and recommend methods
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing analysis results
    """
    try:
        # Get basic dataset info
        n_samples, n_features = df.shape
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Analyze dimensionality
        analysis = analyze_dimensionality(df, n_samples, n_features, numeric_cols, categorical_cols)
        
        # Generate AI insights
        ai_insights = generate_dimensionality_insights(df, analysis)
        
        # Create visualizations
        visualizations = create_dimensionality_visualizations(df, analysis)
        
        # Prepare recommendations
        recommendations = generate_recommendations(analysis, ai_insights)
        
        result = {
            'text': f"Dimensionality Analysis Complete",
            'analysis': analysis,
            'ai_insights': ai_insights,
            'visualizations': visualizations,
            'recommendations': recommendations,
            'success': True
        }
        
        return to_json_serializable(result)
        
    except Exception as e:
        error_result = {
            'text': f"Error in dimensionality reduction analysis: {str(e)}",
            'success': False,
            'error': str(e)
        }
        return to_json_serializable(error_result)

def analyze_dimensionality(df: pd.DataFrame, n_samples: int, n_features: int, 
                          numeric_cols: List[str], categorical_cols: List[str]) -> Dict[str, Any]:
    """Analyze dataset dimensionality characteristics"""
    
    analysis = {
        'n_samples': n_samples,
        'n_features': n_features,
        'n_numeric': len(numeric_cols),
        'n_categorical': len(categorical_cols),
        'dimensionality_ratio': n_features / n_samples if n_samples > 0 else 0,
        'curse_detected': False,
        'recommended_method': None,
        'reasons': []
    }
    
    # Check for dimensionality curse indicators
    if n_features > n_samples:
        analysis['curse_detected'] = True
        analysis['reasons'].append(f"Number of features ({n_features}) exceeds number of samples ({n_samples})")
    
    if n_features > 50:
        analysis['curse_detected'] = True
        analysis['reasons'].append(f"High number of features ({n_features}) may cause overfitting")
    
    if analysis['dimensionality_ratio'] > 0.1:
        analysis['curse_detected'] = True
        analysis['reasons'].append(f"High feature-to-sample ratio ({analysis['dimensionality_ratio']:.3f})")
    
    # Analyze feature correlations for redundancy
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.8:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        if len(high_corr_pairs) > len(numeric_cols) * 0.3:
            analysis['curse_detected'] = True
            analysis['reasons'].append(f"High feature correlation detected ({len(high_corr_pairs)} highly correlated pairs)")
    
    # Recommend method
    if analysis['curse_detected']:
        if n_features > 100 or n_features > n_samples * 2:
            analysis['recommended_method'] = 'PCA'
            analysis['reasons'].append("PCA recommended for high-dimensional data reduction")
        else:
            analysis['recommended_method'] = 't-SNE'
            analysis['reasons'].append("t-SNE recommended for visualization and moderate dimensionality reduction")
    else:
        analysis['recommended_method'] = 'None'
        analysis['reasons'].append("No dimensionality reduction needed")
    
    return analysis

def generate_dimensionality_insights(df: pd.DataFrame, analysis: Dict[str, Any]) -> str:
    """Generate AI insights about dimensionality reduction"""
    
    prompt = f"""
    Analyze this dataset for dimensionality reduction needs:
    
    Dataset Info:
    - Samples: {analysis['n_samples']}
    - Features: {analysis['n_features']}
    - Numeric features: {analysis['n_numeric']}
    - Categorical features: {analysis['n_categorical']}
    - Dimensionality ratio: {analysis['dimensionality_ratio']:.3f}
    - Curse detected: {analysis['curse_detected']}
    - Recommended method: {analysis['recommended_method']}
    
    Reasons: {', '.join(analysis['reasons'])}
    
    Provide insights on:
    1. Whether dimensionality reduction is necessary
    2. The pros and cons of PCA vs t-SNE for this dataset
    3. Expected benefits of applying the recommended method
    4. Alternative approaches if applicable
    """
    
    return generate_insight(prompt)

def create_dimensionality_visualizations(df: pd.DataFrame, analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create visualizations for dimensionality analysis"""
    
    visualizations = {}
    
    # Feature count visualization
    fig_counts = go.Figure()
    fig_counts.add_trace(go.Bar(
        x=['Total Features', 'Numeric', 'Categorical'],
        y=[analysis['n_features'], analysis['n_numeric'], analysis['n_categorical']],
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
    ))
    fig_counts.update_layout(
        title="Feature Distribution",
        xaxis_title="Feature Type",
        yaxis_title="Count",
        showlegend=False
    )
    visualizations['feature_counts'] = fig_counts.to_dict()
    
    # Dimensionality ratio visualization
    fig_ratio = go.Figure()
    fig_ratio.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=analysis['dimensionality_ratio'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Feature-to-Sample Ratio"},
        delta={'reference': 0.1},
        gauge={
            'axis': {'range': [None, 0.5]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.05], 'color': "lightgray"},
                {'range': [0.05, 0.1], 'color': "yellow"},
                {'range': [0.1, 0.5], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.1
            }
        }
    ))
    visualizations['dimensionality_ratio'] = fig_ratio.to_dict()
    
    # Correlation heatmap for numeric features
    if analysis['n_numeric'] > 1:
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        fig_corr.update_layout(
            xaxis_title="Features",
            yaxis_title="Features"
        )
        visualizations['correlation_heatmap'] = fig_corr.to_dict()
    
    return visualizations

def generate_recommendations(analysis: Dict[str, Any], ai_insights: str) -> Dict[str, Any]:
    """Generate specific recommendations for dimensionality reduction"""
    
    recommendations = {
        'needed': analysis['curse_detected'],
        'method': analysis['recommended_method'],
        'reasons': analysis['reasons'],
        'ai_insights': ai_insights,
        'next_steps': []
    }
    
    if analysis['curse_detected']:
        if analysis['recommended_method'] == 'PCA':
            recommendations['next_steps'] = [
                "Apply PCA to reduce dimensions while preserving variance",
                "Choose number of components to explain 95% of variance",
                "Consider feature scaling before PCA"
            ]
        elif analysis['recommended_method'] == 't-SNE':
            recommendations['next_steps'] = [
                "Apply t-SNE for visualization and clustering",
                "Use perplexity parameter between 5-50",
                "Consider running multiple times with different random seeds"
            ]
    else:
        recommendations['next_steps'] = [
            "No dimensionality reduction needed",
            "Focus on feature selection if needed",
            "Consider feature engineering for better performance"
        ]
    
    return recommendations

def apply_pca(df: pd.DataFrame, n_components: int = None, explained_variance: float = 0.95) -> Dict[str, Any]:
    """Apply PCA dimensionality reduction"""
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Select numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            return {
                'success': False,
                'error': 'No numeric columns found for PCA'
            }
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Determine number of components
        if n_components is None:
            pca = PCA()
            pca.fit(scaled_data)
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= explained_variance) + 1
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)
        
        # Create result DataFrame
        result_df = pd.DataFrame(
            pca_result,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        # Add back non-numeric columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number])
        if len(non_numeric_cols.columns) > 0:
            result_df = pd.concat([result_df, non_numeric_cols.reset_index(drop=True)], axis=1)
        
        return {
            'success': True,
            'transformed_data': result_df,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'n_components': n_components,
            'original_features': len(numeric_df.columns)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def apply_tsne(df: pd.DataFrame, n_components: int = 2, perplexity: float = 30.0) -> Dict[str, Any]:
    """Apply t-SNE dimensionality reduction"""
    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        
        # Select numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            return {
                'success': False,
                'error': 'No numeric columns found for t-SNE'
            }
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Apply t-SNE
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        tsne_result = tsne.fit_transform(scaled_data)
        
        # Create result DataFrame
        result_df = pd.DataFrame(
            tsne_result,
            columns=[f'tSNE{i+1}' for i in range(n_components)]
        )
        
        # Add back non-numeric columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number])
        if len(non_numeric_cols.columns) > 0:
            result_df = pd.concat([result_df, non_numeric_cols.reset_index(drop=True)], axis=1)
        
        return {
            'success': True,
            'transformed_data': result_df,
            'n_components': n_components,
            'perplexity': perplexity,
            'original_features': len(numeric_df.columns)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
