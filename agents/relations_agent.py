import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from utils.stats_utils import (
    calculate_cramers_v, calculate_mutual_information, 
    calculate_anova_f_score, calculate_vif, pointbiserial_correlation
)

class RelationsAgent:
    """Agent for analyzing relationships between variables"""
    
    def __init__(self):
        self.name = "Relations Agent"
        
    def process(self, context) -> Dict:
        """
        Analyze relationships between variables
        
        Returns:
            Dict with correlation matrices, MI scores, statistical tests, and visualizations
        """
        df = context.df
        results = {
            'correlations': {},
            'mutual_information': {},
            'statistical_tests': {},
            'multicollinearity': {},
            'visualizations': {},
            'relationship_summary': {}
        }
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 1. Numeric-Numeric Relationships (Correlations)
        if len(numeric_cols) >= 2:
            results['correlations'] = self._analyze_correlations(df[numeric_cols])
            results['multicollinearity'] = self._analyze_multicollinearity(df[numeric_cols])
        
        # 2. Categorical-Categorical Relationships (Cramér's V, Chi-square)
        if len(categorical_cols) >= 2:
            results['statistical_tests']['categorical_associations'] = self._analyze_categorical_associations(df[categorical_cols])
        
        # 3. Numeric-Categorical Relationships (ANOVA, Point-biserial)
        if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
            results['statistical_tests']['numeric_categorical'] = self._analyze_numeric_categorical(df, numeric_cols, categorical_cols)
        
        # 4. Mutual Information (universal dependency measure)
        if len(df.columns) >= 2:
            results['mutual_information'] = self._calculate_pairwise_mi(df)
        
        # 5. Generate visualizations
        results['visualizations'] = self._generate_relationship_visualizations(df, results)
        
        # 6. Relationship summary
        results['relationship_summary'] = self._summarize_relationships(results, numeric_cols, categorical_cols)
        
        return results
    
    def _analyze_correlations(self, df_numeric: pd.DataFrame) -> Dict:
        """Analyze correlations between numeric variables"""
        correlations = {}
        
        # Pearson correlation
        pearson_corr = df_numeric.corr(method='pearson')
        correlations['pearson'] = {
            'matrix': pearson_corr,
            'strong_pairs': self._find_strong_correlations(pearson_corr, threshold=0.7)
        }
        
        # Spearman correlation (for monotonic relationships)
        spearman_corr = df_numeric.corr(method='spearman')
        correlations['spearman'] = {
            'matrix': spearman_corr,
            'strong_pairs': self._find_strong_correlations(spearman_corr, threshold=0.7)
        }
        
        return correlations
    
    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
        """Find pairs with strong correlations"""
        strong_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    strong_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val,
                        'strength': 'Very Strong' if abs(corr_val) >= 0.9 else 'Strong'
                    })
        
        return sorted(strong_pairs, key=lambda x: abs(x['correlation']), reverse=True)
    
    def _analyze_multicollinearity(self, df_numeric: pd.DataFrame) -> Dict:
        """Analyze multicollinearity using VIF"""
        try:
            vif_df = calculate_vif(df_numeric)
            
            return {
                'vif_scores': vif_df,
                'high_vif_features': vif_df[vif_df['VIF'] > 10]['Feature'].tolist() if not vif_df.empty else [],
                'recommendations': self._get_multicollinearity_recommendations(vif_df)
            }
        except:
            return {'vif_scores': pd.DataFrame(), 'high_vif_features': [], 'recommendations': []}
    
    def _get_multicollinearity_recommendations(self, vif_df: pd.DataFrame) -> List[str]:
        """Generate recommendations for multicollinearity issues"""
        recommendations = []
        
        if vif_df.empty:
            return recommendations
            
        high_vif = vif_df[vif_df['VIF'] > 10]
        if not high_vif.empty:
            recommendations.append(f"Consider removing features with high VIF: {high_vif['Feature'].tolist()}")
            recommendations.append("Alternative: Use regularization (Ridge/Lasso) to handle multicollinearity")
            recommendations.append("Consider PCA or feature selection techniques")
        
        return recommendations
    
    def _analyze_categorical_associations(self, df_categorical: pd.DataFrame) -> Dict:
        """Analyze associations between categorical variables"""
        associations = {}
        cramers_v_matrix = pd.DataFrame(index=df_categorical.columns, columns=df_categorical.columns)
        
        for col1 in df_categorical.columns:
            for col2 in df_categorical.columns:
                if col1 == col2:
                    cramers_v_matrix.loc[col1, col2] = 1.0
                else:
                    cramers_v = calculate_cramers_v(df_categorical[col1].fillna('Missing'), 
                                                  df_categorical[col2].fillna('Missing'))
                    cramers_v_matrix.loc[col1, col2] = cramers_v
        
        # Convert to numeric
        cramers_v_matrix = cramers_v_matrix.astype(float)
        
        associations['cramers_v_matrix'] = cramers_v_matrix
        associations['strong_associations'] = self._find_strong_associations(cramers_v_matrix, threshold=0.5)
        
        return associations
    
    def _find_strong_associations(self, association_matrix: pd.DataFrame, threshold: float = 0.5) -> List[Dict]:
        """Find strong categorical associations"""
        strong_pairs = []
        
        for i in range(len(association_matrix.columns)):
            for j in range(i+1, len(association_matrix.columns)):
                assoc_val = association_matrix.iloc[i, j]
                if assoc_val >= threshold:
                    strong_pairs.append({
                        'var1': association_matrix.columns[i],
                        'var2': association_matrix.columns[j],
                        'cramers_v': assoc_val,
                        'strength': 'Very Strong' if assoc_val >= 0.8 else 'Strong'
                    })
        
        return sorted(strong_pairs, key=lambda x: x['cramers_v'], reverse=True)
    
    def _analyze_numeric_categorical(self, df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> Dict:
        """Analyze relationships between numeric and categorical variables"""
        results = {}
        anova_results = []
        
        for num_col in numeric_cols:
            for cat_col in categorical_cols:
                # ANOVA F-test
                f_stat, p_value = calculate_anova_f_score(df[num_col], df[cat_col])
                
                # Point-biserial for binary categorical
                point_biserial = None
                if df[cat_col].nunique() == 2:
                    # Convert to binary numeric
                    binary_encoded = pd.get_dummies(df[cat_col], drop_first=True).iloc[:, 0] if df[cat_col].nunique() == 2 else None
                    if binary_encoded is not None:
                        point_biserial = pointbiserial_correlation(binary_encoded, df[num_col])
                
                anova_results.append({
                    'numeric_var': num_col,
                    'categorical_var': cat_col,
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'point_biserial': point_biserial
                })
        
        results['anova_tests'] = sorted(anova_results, key=lambda x: x['f_statistic'], reverse=True)
        results['significant_relationships'] = [r for r in anova_results if r['significant']]
        
        return results
    
    def _calculate_pairwise_mi(self, df: pd.DataFrame) -> Dict:
        """Calculate pairwise mutual information"""
        mi_results = {}
        
        # For each pair of columns, calculate MI
        mi_matrix = pd.DataFrame(index=df.columns, columns=df.columns)
        
        for col1 in df.columns:
            for col2 in df.columns:
                if col1 == col2:
                    mi_matrix.loc[col1, col2] = 0.0  # MI with itself is not informative
                else:
                    try:
                        # Prepare data
                        X = df[[col1]].fillna(df[col1].mode()[0] if df[col1].dtype == 'object' else df[col1].median())
                        y = df[col2].fillna(df[col2].mode()[0] if df[col2].dtype == 'object' else df[col2].median())
                        
                        # Calculate MI
                        mi_score = calculate_mutual_information(X, y)[col1]
                        mi_matrix.loc[col1, col2] = mi_score
                    except:
                        mi_matrix.loc[col1, col2] = 0.0
        
        mi_matrix = mi_matrix.astype(float)
        
        mi_results['matrix'] = mi_matrix
        mi_results['top_dependencies'] = self._find_top_mi_pairs(mi_matrix)
        
        return mi_results
    
    def _find_top_mi_pairs(self, mi_matrix: pd.DataFrame, top_n: int = 10) -> List[Dict]:
        """Find top mutual information pairs"""
        pairs = []
        
        for i in range(len(mi_matrix.columns)):
            for j in range(i+1, len(mi_matrix.columns)):
                mi_val = mi_matrix.iloc[i, j]
                pairs.append({
                    'var1': mi_matrix.columns[i],
                    'var2': mi_matrix.columns[j],
                    'mutual_information': mi_val
                })
        
        return sorted(pairs, key=lambda x: x['mutual_information'], reverse=True)[:top_n]
    
    def _generate_relationship_visualizations(self, df: pd.DataFrame, results: Dict) -> Dict:
        """Generate visualizations for relationships"""
        viz = {}
        
        # Correlation heatmaps
        if 'correlations' in results and results['correlations']:
            viz['correlation_heatmaps'] = self._plot_correlation_heatmaps(results['correlations'])
        
        # Mutual information heatmap
        if 'mutual_information' in results and 'matrix' in results['mutual_information']:
            viz['mi_heatmap'] = self._plot_mi_heatmap(results['mutual_information']['matrix'])
        
        # Categorical associations heatmap
        if 'statistical_tests' in results and 'categorical_associations' in results['statistical_tests']:
            cat_assoc = results['statistical_tests']['categorical_associations']
            if 'cramers_v_matrix' in cat_assoc:
                viz['cramers_v_heatmap'] = self._plot_cramers_v_heatmap(cat_assoc['cramers_v_matrix'])
        
        return viz
    
    def _plot_correlation_heatmaps(self, correlations: Dict) -> Dict:
        """Create correlation heatmap visualizations"""
        heatmaps = {}
        
        for corr_type, corr_data in correlations.items():
            if 'matrix' in corr_data:
                fig = px.imshow(
                    corr_data['matrix'], 
                    color_continuous_scale='RdBu_r',
                    aspect='auto',
                    title=f'{corr_type.title()} Correlation Matrix',
                    zmin=-1, zmax=1
                )
                fig.update_layout(height=500)
                heatmaps[f'{corr_type}_heatmap'] = fig
        
        return heatmaps
    
    def _plot_mi_heatmap(self, mi_matrix: pd.DataFrame):
        """Create mutual information heatmap"""
        fig = px.imshow(
            mi_matrix,
            color_continuous_scale='Viridis',
            aspect='auto',
            title='Mutual Information Matrix'
        )
        fig.update_layout(height=500)
        return fig
    
    def _plot_cramers_v_heatmap(self, cramers_matrix: pd.DataFrame):
        """Create Cramér's V heatmap"""
        fig = px.imshow(
            cramers_matrix,
            color_continuous_scale='Blues',
            aspect='auto',
            title="Cramér's V Association Matrix (Categorical Variables)",
            zmin=0, zmax=1
        )
        fig.update_layout(height=500)
        return fig
    
    def _summarize_relationships(self, results: Dict, numeric_cols: List[str], categorical_cols: List[str]) -> Dict:
        """Summarize key relationship findings"""
        summary = {
            'total_features': len(numeric_cols) + len(categorical_cols),
            'numeric_features': len(numeric_cols),
            'categorical_features': len(categorical_cols),
            'key_findings': [],
            'recommendations': []
        }
        
        # Strong correlations
        if 'correlations' in results:
            for corr_type, corr_data in results['correlations'].items():
                strong_pairs = corr_data.get('strong_pairs', [])
                if strong_pairs:
                    summary['key_findings'].append(f"{len(strong_pairs)} strong {corr_type} correlations found")
        
        # Multicollinearity
        if 'multicollinearity' in results:
            high_vif = results['multicollinearity'].get('high_vif_features', [])
            if high_vif:
                summary['key_findings'].append(f"Multicollinearity detected in {len(high_vif)} features")
                summary['recommendations'].extend(results['multicollinearity'].get('recommendations', []))
        
        # Categorical associations
        if 'statistical_tests' in results and 'categorical_associations' in results['statistical_tests']:
            strong_assoc = results['statistical_tests']['categorical_associations'].get('strong_associations', [])
            if strong_assoc:
                summary['key_findings'].append(f"{len(strong_assoc)} strong categorical associations found")
        
        # Significant numeric-categorical relationships
        if 'statistical_tests' in results and 'numeric_categorical' in results['statistical_tests']:
            significant = results['statistical_tests']['numeric_categorical'].get('significant_relationships', [])
            if significant:
                summary['key_findings'].append(f"{len(significant)} significant numeric-categorical relationships")
        
        return summary