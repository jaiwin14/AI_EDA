import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import re
from utils.stats_utils import calculate_mutual_information
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class TargetTaskAgent:
    """Agent for automatically detecting target column and inferring task type"""
    
    def __init__(self):
        self.name = "Target & Task Agent"
        
    def process(self, context) -> Dict:
        """
        Detect target column and infer task type
        
        Returns:
            Dict with target detection results and task inference
        """
        df = context.df
        results = {
            'target_candidates': [],
            'recommended_target': None,
            'task_type': None,
            'confidence_score': 0.0,
            'reasoning': [],
            'target_analysis': {}
        }
        
        # Get all target candidates with scores
        candidates = self._score_target_candidates(df)
        results['target_candidates'] = candidates
        
        if candidates:
            # Select best candidate
            best_candidate = candidates[0]  # Already sorted by score
            results['recommended_target'] = best_candidate['column']
            results['confidence_score'] = best_candidate['score']
            results['reasoning'] = best_candidate['reasoning']
            
            # Infer task type
            target_col = best_candidate['column']
            task_info = self._infer_task_type(df[target_col])
            results['task_type'] = task_info['task_type']
            results['target_analysis'] = task_info
            
            # Update context
            context.target_column = target_col
            context.task_type = task_info['task_type']
        
        return results
    
    def get_target_override_options(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Get target override options based on the dataframe columns.
        Returns a list of dictionaries with column info for override selection,
        sorted by their suitability as targets.
        """
        # Get all target candidates with their scores
        candidates = self._score_target_candidates(df)
        
        # Build enhanced options with additional metadata
        target_options = []
        
        for candidate in candidates:
            col = candidate['column']
            series = df[col]
            
            # Skip columns that are clearly not suitable as targets
            skip_column = False
            
            # Skip ID-like columns (too many unique values)
            if series.nunique() > 0.95 * len(series) and len(series) > 100:
                skip_column = True
            
            # Skip constant columns
            elif series.nunique() <= 1:
                skip_column = True
            
            # Skip columns with extremely high missing data
            elif series.isnull().sum() / len(series) > 0.8:
                skip_column = True
            
            # Skip obvious identifier columns by name
            elif col.lower() in ['index', 'id', 'key', 'row_id', 'record_id', 'uid', 'guid']:
                skip_column = True
            
            if not skip_column:
                # Create enhanced option with metadata
                option = {
                    'column': col,
                    'dtype': str(series.dtype),
                    'unique_count': series.nunique(),
                    'missing_count': series.isnull().sum(),
                    'missing_ratio': series.isnull().sum() / len(series),
                    'score': candidate['score'],
                    'reasoning': candidate['reasoning'][:3]  # Top 3 reasons
                }
                target_options.append(option)
        
        # If filtering removed everything, create basic options from all columns
        if not target_options:
            for col in df.columns:
                series = df[col]
                option = {
                    'column': col,
                    'dtype': str(series.dtype),
                    'unique_count': series.nunique(),
                    'missing_count': series.isnull().sum(),
                    'missing_ratio': series.isnull().sum() / len(series),
                    'score': 0.0,
                    'reasoning': ['Fallback option - no suitable candidates found']
                }
                target_options.append(option)
        
        return target_options
    
    def _score_target_candidates(self, df: pd.DataFrame) -> List[Dict]:
        """
        Score all columns as potential targets based on various heuristics
        
        Returns:
            List of candidate dictionaries sorted by score (highest first)
        """
        candidates = []
        
        for col in df.columns:
            score = 0.0
            reasoning = []
            
            # 1. Name-based scoring (strongest signal)
            name_score, name_reasons = self._score_by_name(col)
            score += name_score
            reasoning.extend(name_reasons)
            
            # 2. Position-based scoring
            position_score, position_reasons = self._score_by_position(col, df.columns)
            score += position_score
            reasoning.extend(position_reasons)
            
            # 3. Statistical properties scoring
            stats_score, stats_reasons = self._score_by_statistics(df[col])
            score += stats_score
            reasoning.extend(stats_reasons)
            
            # 4. Predictability scoring (expensive, so only for top candidates by name/position)
            if score > 10:  # Only calculate for promising candidates
                pred_score, pred_reasons = self._score_by_predictability(df, col)
                score += pred_score
                reasoning.extend(pred_reasons)
            
            # 5. Penalties for obvious non-targets
            penalty_score, penalty_reasons = self._apply_penalties(df[col], col)
            score += penalty_score
            reasoning.extend(penalty_reasons)
            
            candidates.append({
                'column': col,
                'score': max(0, score),  # Don't allow negative scores
                'reasoning': reasoning
            })
        
        # Sort by score (highest first)
        return sorted(candidates, key=lambda x: x['score'], reverse=True)
    
    def _score_by_name(self, col_name: str) -> Tuple[float, List[str]]:
        """Score column based on name patterns"""
        score = 0.0
        reasoning = []
        col_lower = col_name.lower().strip()
        
        # Strong target indicators
        strong_patterns = [
            'target', 'label', 'class', 'outcome', 'response', 'dependent',
            'prediction', 'predict', 'output', 'result', 'goal', 'objective'
        ]
        
        for pattern in strong_patterns:
            if pattern in col_lower:
                score += 50
                reasoning.append(f"Strong name indicator: contains '{pattern}'")
                break
        
        # Moderate target indicators
        moderate_patterns = [
            'y', 'category', 'type', 'status', 'flag', 'indicator'
        ]
        
        for pattern in moderate_patterns:
            if col_lower == pattern or col_lower.endswith('_' + pattern):
                score += 30
                reasoning.append(f"Moderate name indicator: '{pattern}'")
                break
        
        # Weak target indicators (common suffixes/prefixes)
        if col_lower.startswith('is_') or col_lower.startswith('has_'):
            score += 15
            reasoning.append("Weak indicator: boolean-like prefix")
        
        if col_lower.endswith('_flag') or col_lower.endswith('_ind'):
            score += 15
            reasoning.append("Weak indicator: flag/indicator suffix")
        
        return score, reasoning
    
    def _score_by_position(self, col_name: str, all_columns: List[str]) -> Tuple[float, List[str]]:
        """Score based on column position (last column often target)"""
        score = 0.0
        reasoning = []
        
        col_index = list(all_columns).index(col_name)
        
        # Last column bonus
        if col_index == len(all_columns) - 1:
            score += 20
            reasoning.append("Position bonus: last column")
        
        # Second to last gets smaller bonus
        elif col_index == len(all_columns) - 2:
            score += 10
            reasoning.append("Position bonus: second to last column")
        
        return score, reasoning
    
    def _score_by_statistics(self, series: pd.Series) -> Tuple[float, List[str]]:
        """Score based on statistical properties of the column"""
        score = 0.0
        reasoning = []
        
        # Check cardinality for classification potential
        unique_count = series.nunique()
        total_count = len(series)
        unique_ratio = unique_count / total_count if total_count > 0 else 0
        
        # Good classification target: low-medium cardinality
        if 2 <= unique_count <= min(50, total_count * 0.2):
            if unique_count <= 10:
                score += 25
                reasoning.append(f"Good classification target: {unique_count} unique values")
            else:
                score += 15
                reasoning.append(f"Possible classification target: {unique_count} unique values")
        
        # Good regression target: high cardinality numeric
        elif series.dtype in ['int64', 'float64'] and unique_ratio > 0.2:
            score += 20
            reasoning.append(f"Good regression target: continuous numeric with {unique_count} unique values")
        
        # Binary target bonus
        if unique_count == 2:
            score += 15
            reasoning.append("Binary target bonus")
        
        # Check for reasonable missing data
        missing_ratio = series.isnull().sum() / len(series)
        if missing_ratio > 0.3:
            score -= 20
            reasoning.append(f"Penalty: high missing data ({missing_ratio:.1%})")
        elif missing_ratio == 0:
            score += 5
            reasoning.append("Bonus: no missing values")
        
        return score, reasoning
    
    def _score_by_predictability(self, df: pd.DataFrame, target_col: str) -> Tuple[float, List[str]]:
        """Score based on how well other features can predict this column"""
        score = 0.0
        reasoning = []
        
        try:
            # Prepare features (all columns except the potential target)
            feature_cols = [col for col in df.columns if col != target_col]
            if not feature_cols:
                return 0.0, ["No features available for predictability analysis"]
            
            X = df[feature_cols]
            y = df[target_col]
            
            # Calculate mutual information with all features
            mi_scores = calculate_mutual_information(X, y)
            
            # Average MI score as predictability measure
            avg_mi = mi_scores.mean()
            max_mi = mi_scores.max()
            
            # Score based on predictability
            if avg_mi > 0.1:
                score += min(avg_mi * 100, 30)
                reasoning.append(f"High predictability: avg MI = {avg_mi:.3f}")
            
            if max_mi > 0.3:
                score += 10
                reasoning.append(f"Strong single predictor found: max MI = {max_mi:.3f}")
                
        except Exception as e:
            reasoning.append(f"Predictability analysis failed: {str(e)}")
        
        return score, reasoning
    
    def _apply_penalties(self, series: pd.Series, col_name: str) -> Tuple[float, List[str]]:
        """Apply penalties for columns unlikely to be targets"""
        penalty = 0.0
        reasoning = []
        
        # ID-like columns
        if series.nunique() > 0.95 * len(series) and len(series) > 100:
            penalty -= 50
            reasoning.append("Major penalty: ID-like column (too many unique values)")
        
        # Timestamp/date columns
        if 'date' in col_name.lower() or 'time' in col_name.lower():
            penalty -= 30
            reasoning.append("Penalty: likely timestamp column")
        
        # Index-like columns
        if col_name.lower() in ['index', 'id', 'key', 'row_id', 'record_id']:
            penalty -= 40
            reasoning.append("Penalty: index-like column name")
        
        # Constant columns
        if series.nunique() <= 1:
            penalty -= 100
            reasoning.append("Major penalty: constant column")
        
        # Text-heavy columns (likely not targets)
        if series.dtype == 'object':
            avg_length = series.astype(str).str.len().mean()
            if avg_length > 50:  # Long text fields
                penalty -= 25
                reasoning.append("Penalty: long text column (likely descriptive)")
        
        return penalty, reasoning
    
    def _infer_task_type(self, target_series: pd.Series) -> Dict[str, Any]:
        """
        Infer the machine learning task type based on target column
        
        Returns:
            Dict with task type and analysis details
        """
        analysis = {
            'task_type': None,
            'target_dtype': str(target_series.dtype),
            'unique_values': target_series.nunique(),
            'unique_ratio': target_series.nunique() / len(target_series),
            'sample_values': target_series.dropna().unique()[:10].tolist(),
            'missing_count': target_series.isnull().sum(),
            'missing_ratio': target_series.isnull().sum() / len(target_series),
            'reasoning': []
        }
        
        unique_count = analysis['unique_values']
        unique_ratio = analysis['unique_ratio']
        
        # Classification criteria
        if (target_series.dtype in ['object', 'category'] or 
            (target_series.dtype in ['int64', 'float64'] and unique_count <= min(20, len(target_series) * 0.05))):
            
            analysis['task_type'] = 'classification'
            
            if unique_count == 2:
                analysis['classification_type'] = 'binary'
                analysis['reasoning'].append("Binary classification: exactly 2 unique values")
            else:
                analysis['classification_type'] = 'multiclass'
                analysis['reasoning'].append(f"Multiclass classification: {unique_count} classes")
            
            # Class balance analysis
            value_counts = target_series.value_counts()
            analysis['class_distribution'] = value_counts.to_dict()
            
            # Check for imbalance
            majority_ratio = value_counts.iloc[0] / len(target_series)
            if majority_ratio > 0.9:
                analysis['class_balance'] = 'severely_imbalanced'
                analysis['reasoning'].append(f"Severely imbalanced: {majority_ratio:.1%} majority class")
            elif majority_ratio > 0.7:
                analysis['class_balance'] = 'imbalanced'
                analysis['reasoning'].append(f"Imbalanced: {majority_ratio:.1%} majority class")
            else:
                analysis['class_balance'] = 'balanced'
                analysis['reasoning'].append("Reasonably balanced classes")
        
        # Regression criteria
        elif target_series.dtype in ['int64', 'float64'] and unique_ratio > 0.05:
            analysis['task_type'] = 'regression'
            analysis['reasoning'].append(f"Regression: continuous numeric with {unique_count} unique values")
            
            # Additional regression analysis
            analysis['target_stats'] = {
                'mean': target_series.mean(),
                'std': target_series.std(),
                'min': target_series.min(),
                'max': target_series.max(),
                'skewness': target_series.skew(),
                'kurtosis': target_series.kurtosis()
            }
            
            # Check distribution characteristics
            skewness = analysis['target_stats']['skewness']
            if abs(skewness) > 2:
                analysis['distribution_note'] = 'highly_skewed'
                analysis['reasoning'].append(f"Highly skewed distribution (skew={skewness:.2f})")
            elif abs(skewness) > 1:
                analysis['distribution_note'] = 'moderately_skewed'
                analysis['reasoning'].append(f"Moderately skewed distribution (skew={skewness:.2f})")
            else:
                analysis['distribution_note'] = 'approximately_normal'
                analysis['reasoning'].append("Approximately normal distribution")
        
        else:
            # Ambiguous case
            analysis['task_type'] = 'unclear'
            analysis['reasoning'].append("Cannot clearly determine task type - may need manual specification")
            
            if unique_ratio < 0.05:
                analysis['reasoning'].append("Low unique ratio suggests classification")
            else:
                analysis['reasoning'].append("High unique ratio suggests regression")
        
        return analysis

    def suggest_target_override(self, df: pd.DataFrame, manual_target: str) -> Dict[str, Any]:
        """
        Analyze manually specified target and provide feedback
        
        Args:
            df: DataFrame
            manual_target: Column name specified by user
            
        Returns:
            Analysis of the manual target choice
        """
        if manual_target not in df.columns:
            return {
                'valid': False,
                'error': f"Column '{manual_target}' not found in dataset",
                'suggestions': []
            }
        
        target_series = df[manual_target]
        task_analysis = self._infer_task_type(target_series)
        
        # Get automatic recommendation for comparison
        auto_results = self.process(type('Context', (), {'df': df})())
        auto_target = auto_results.get('recommended_target')
        
        analysis = {
            'valid': True,
            'manual_target': manual_target,
            'task_analysis': task_analysis,
            'comparison_with_auto': {
                'auto_recommendation': auto_target,
                'matches_auto': manual_target == auto_target,
                'manual_choice_reasoning': []
            },
            'warnings': [],
            'recommendations': []
        }
        
        # Generate warnings and recommendations
        if task_analysis['missing_ratio'] > 0.3:
            analysis['warnings'].append(f"High missing data in target: {task_analysis['missing_ratio']:.1%}")
        
        if manual_target != auto_target and auto_target:
            analysis['comparison_with_auto']['manual_choice_reasoning'].append(
                f"Manual choice differs from auto-detected target '{auto_target}'"
            )
        
        # Task-specific recommendations
        if task_analysis['task_type'] == 'classification':
            if analysis.get('class_balance') == 'severely_imbalanced':
                analysis['recommendations'].append("Consider using stratified sampling and balanced metrics")
                analysis['recommendations'].append("Use SMOTE or class weights to handle imbalance")
        
        elif task_analysis['task_type'] == 'regression':
            if task_analysis.get('distribution_note') == 'highly_skewed':
                analysis['recommendations'].append("Consider log transformation for skewed target")
        
        return analysis