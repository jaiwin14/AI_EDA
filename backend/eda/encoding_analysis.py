"""Categorical Encoding Analysis Module with AI Recommendations"""
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
from utils.serialization import to_json_serializable

def run(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the need for categorical encoding and recommend appropriate methods
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing encoding analysis and recommendations
    """
    try:
        # Select categorical columns only
        categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        if not categorical_columns:
            return {
                "text": "No categorical columns found for encoding analysis",
                "error": "No categorical columns available"
            }
        
        results = {
            "text": f"Encoding analysis completed for {len(categorical_columns)} categorical columns",
            "column_analysis": {},
            "overall_recommendation": {},
            "ai_insights": "",
            "visualizations": {}
        }
        
        # Analyze each categorical column
        for column in categorical_columns:
            col_analysis = analyze_column_encoding(df[column], column)
            results["column_analysis"][column] = col_analysis
        
        # Generate overall recommendation
        results["overall_recommendation"] = generate_overall_encoding_recommendation(results["column_analysis"])
        
        # Generate AI insights
        insight_prompt = f"""
        Analyze the categorical encoding analysis results for this dataset:
        - Total categorical columns: {len(categorical_columns)}
        - Columns needing encoding: {sum(1 for col in results['column_analysis'].values() if col.get('needs_encoding', False))}
        
        For each column that needs encoding, provide insights on:
        1. Why encoding is recommended (algorithm requirements, cardinality issues, etc.)
        2. Which method is most appropriate (Label Encoding vs One-Hot Encoding) and why
        3. Potential impact on machine learning models
        4. Business context considerations
        5. Implementation recommendations
        
        Explain the differences between Label Encoding and One-Hot Encoding:
        - Label Encoding: Converts categories to integers (0, 1, 2, ...)
        - One-Hot Encoding: Creates binary columns for each category
        
        Consider factors like:
        - Cardinality (number of unique values)
        - Ordinal vs nominal relationships
        - Algorithm requirements (some algorithms assume ordinal relationships)
        - Dimensionality impact
        - Interpretability needs
        """
        
        try:
            results["ai_insights"] = generate_insight(insight_prompt, context=results["column_analysis"])
        except Exception as e:
            results["ai_insights"] = f"Unable to generate AI insights: {str(e)}"
        
        # Create visualizations
        results["visualizations"] = create_encoding_visualizations(df, results["column_analysis"])
        
        return results
        
    except Exception as e:
        return {
            "text": f"Error in encoding analysis: {str(e)}",
            "error": str(e)
        }

def analyze_column_encoding(series: pd.Series, column_name: str) -> Dict[str, Any]:
    """Analyze encoding needs for a single categorical column"""
    
    # Remove nulls for analysis
    clean_series = series.dropna()
    
    if len(clean_series) == 0:
        return {"error": "No valid categorical data"}
    
    # Basic statistics
    unique_count = int(clean_series.nunique())
    total_count = int(len(clean_series))
    cardinality_ratio = unique_count / total_count if total_count > 0 else 0
    
    # Value counts
    value_counts = clean_series.value_counts()
    value_counts_percent = clean_series.value_counts(normalize=True) * 100
    
    # Check for ordinal relationships
    is_ordinal = check_ordinal_relationship(clean_series)
    
    # Check for high cardinality
    is_high_cardinality = unique_count > 50  # Arbitrary threshold
    
    # Check for balanced vs imbalanced categories
    most_common_percentage = value_counts_percent.iloc[0] if len(value_counts_percent) > 0 else 0
    is_imbalanced = most_common_percentage > 80  # If one category dominates
    
    # Determine if encoding is needed
    needs_encoding = determine_encoding_need(unique_count, cardinality_ratio, is_ordinal)
    
    # Recommend appropriate method
    recommended_method = recommend_encoding_method(
        unique_count, cardinality_ratio, is_ordinal, is_high_cardinality, is_imbalanced
    )
    
    analysis = {
        "column_name": column_name,
        "statistics": {
            "total_count": total_count,
            "unique_count": unique_count,
            "cardinality_ratio": round(cardinality_ratio, 3),
            "most_common_value": value_counts.index[0] if len(value_counts) > 0 else None,
            "most_common_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            "most_common_percentage": round(most_common_percentage, 2),
            "least_common_value": value_counts.index[-1] if len(value_counts) > 0 else None,
            "least_common_count": int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
            "least_common_percentage": round(value_counts_percent.iloc[-1], 2) if len(value_counts_percent) > 0 else 0
        },
        "characteristics": {
            "is_ordinal": is_ordinal,
            "is_high_cardinality": is_high_cardinality,
            "is_imbalanced": is_imbalanced,
            "value_distribution": value_counts.to_dict(),
            "value_distribution_percent": value_counts_percent.to_dict()
        },
        "encoding_assessment": {
            "needs_encoding": needs_encoding,
            "recommended_method": recommended_method,
            "reasoning": generate_encoding_reasoning(
                unique_count, cardinality_ratio, is_ordinal, is_high_cardinality, 
                is_imbalanced, recommended_method
            )
        }
    }
    
    return analysis

def check_ordinal_relationship(series: pd.Series) -> bool:
    """Check if categorical values have ordinal relationships"""
    
    # Common ordinal patterns
    ordinal_patterns = [
        # Education levels
        ['elementary', 'middle', 'high', 'bachelor', 'master', 'phd'],
        ['primary', 'secondary', 'tertiary'],
        ['low', 'medium', 'high'],
        ['poor', 'fair', 'good', 'excellent'],
        ['never', 'rarely', 'sometimes', 'often', 'always'],
        ['disagree', 'neutral', 'agree'],
        ['small', 'medium', 'large'],
        ['beginner', 'intermediate', 'advanced'],
        ['junior', 'senior', 'lead', 'manager'],
        ['cold', 'warm', 'hot'],
        ['slow', 'fast'],
        ['cheap', 'expensive'],
        ['easy', 'medium', 'hard'],
        ['low', 'medium', 'high'],
        ['minimal', 'moderate', 'extensive'],
        ['none', 'low', 'medium', 'high'],
        ['bad', 'average', 'good', 'excellent'],
        ['unlikely', 'possible', 'likely', 'certain'],
        ['weak', 'moderate', 'strong'],
        ['minor', 'major', 'critical']
    ]
    
    # Check if series values match any ordinal pattern
    unique_values = set(series.unique())
    
    for pattern in ordinal_patterns:
        pattern_set = set(pattern)
        if unique_values.issubset(pattern_set) and len(unique_values) >= 2:
            return True
    
    # Check for numeric-like strings that might be ordinal
    numeric_like = 0
    for value in unique_values:
        if isinstance(value, str):
            # Check for common ordinal indicators
            if any(indicator in value.lower() for indicator in ['1st', '2nd', '3rd', 'first', 'second', 'third', 'level', 'grade', 'class']):
                numeric_like += 1
            # Check if it's a number
            elif value.replace('.', '').replace('-', '').isdigit():
                numeric_like += 1
    
    # If more than 50% of values are numeric-like, consider it ordinal
    return numeric_like > len(unique_values) * 0.5

def determine_encoding_need(unique_count: int, cardinality_ratio: float, is_ordinal: bool) -> bool:
    """Determine if encoding is needed based on various factors"""
    
    # Always need encoding for categorical data in ML
    # But we can provide recommendations on the best approach
    
    # Multiple criteria for determining encoding strategy
    criteria = []
    
    # 1. High cardinality
    if unique_count > 20:
        criteria.append("high_cardinality")
    
    # 2. High cardinality ratio
    if cardinality_ratio > 0.5:
        criteria.append("high_cardinality_ratio")
    
    # 3. Ordinal relationship
    if is_ordinal:
        criteria.append("ordinal_relationship")
    
    # 4. Binary categories
    if unique_count == 2:
        criteria.append("binary_categories")
    
    # Encoding is always needed, but the method depends on these factors
    return True

def recommend_encoding_method(unique_count: int, cardinality_ratio: float, is_ordinal: bool,
                            is_high_cardinality: bool, is_imbalanced: bool) -> str:
    """Recommend appropriate encoding method"""
    
    # Factors favoring Label Encoding
    label_encoding_factors = 0
    
    # 1. Ordinal relationship
    if is_ordinal:
        label_encoding_factors += 2  # Strong factor
    
    # 2. Low cardinality
    if unique_count <= 10:
        label_encoding_factors += 1
    
    # 3. Tree-based algorithms (assume ordinal relationships)
    label_encoding_factors += 1  # General consideration
    
    # 4. Memory efficiency
    label_encoding_factors += 1
    
    # Factors favoring One-Hot Encoding
    onehot_encoding_factors = 0
    
    # 1. Nominal categories (no ordinal relationship)
    if not is_ordinal:
        onehot_encoding_factors += 2  # Strong factor
    
    # 2. High cardinality (but not too high)
    if 5 < unique_count <= 20:
        onehot_encoding_factors += 1
    
    # 3. Linear models (sensitive to ordinal assumptions)
    onehot_encoding_factors += 1
    
    # 4. Interpretability
    onehot_encoding_factors += 1
    
    # Special cases
    if unique_count == 2:
        # Binary categories can use either, but label encoding is simpler
        return "Label Encoding"
    elif unique_count > 50:
        # Very high cardinality - consider target encoding or feature hashing
        return "Target Encoding" if is_imbalanced else "Feature Hashing"
    
    # Decision logic
    if label_encoding_factors > onehot_encoding_factors:
        return "Label Encoding"
    elif onehot_encoding_factors > label_encoding_factors:
        return "One-Hot Encoding"
    else:
        # Tie-breaker: prefer one-hot for nominal data
        return "One-Hot Encoding" if not is_ordinal else "Label Encoding"

def generate_encoding_reasoning(unique_count: int, cardinality_ratio: float, is_ordinal: bool,
                              is_high_cardinality: bool, is_imbalanced: bool, method: str) -> str:
    """Generate reasoning for encoding recommendation"""
    
    reasons = []
    
    if method == "Label Encoding":
        reasons.append("Label Encoding is recommended because:")
        if is_ordinal:
            reasons.append("- Categories have ordinal relationships")
        if unique_count <= 10:
            reasons.append("- Low cardinality (few unique values)")
        reasons.append("- Memory efficient")
        reasons.append("- Good for tree-based algorithms")
        reasons.append("- Preserves ordinal relationships")
        
    elif method == "One-Hot Encoding":
        reasons.append("One-Hot Encoding is recommended because:")
        if not is_ordinal:
            reasons.append("- Categories are nominal (no ordinal relationship)")
        if 5 < unique_count <= 20:
            reasons.append("- Moderate cardinality")
        reasons.append("- Good for linear models")
        reasons.append("- Highly interpretable")
        reasons.append("- No ordinal assumptions")
        
    elif method == "Target Encoding":
        reasons.append("Target Encoding is recommended because:")
        reasons.append("- Very high cardinality")
        if is_imbalanced:
            reasons.append("- Imbalanced categories")
        reasons.append("- Reduces dimensionality")
        reasons.append("- Captures target relationship")
        
    elif method == "Feature Hashing":
        reasons.append("Feature Hashing is recommended because:")
        reasons.append("- Extremely high cardinality")
        reasons.append("- Memory efficient")
        reasons.append("- Fixed dimensionality output")
        
    return " ".join(reasons)

def generate_overall_encoding_recommendation(column_analysis: Dict) -> Dict[str, Any]:
    """Generate overall encoding recommendation"""
    
    total_columns = len(column_analysis)
    columns_needing_encoding = sum(
        1 for col in column_analysis.values() 
        if col.get('encoding_assessment', {}).get('needs_encoding', False)
    )
    
    method_counts = {}
    ordinal_count = 0
    high_cardinality_count = 0
    
    for col in column_analysis.values():
        if 'encoding_assessment' in col:
            method = col['encoding_assessment'].get('recommended_method', 'None')
            method_counts[method] = method_counts.get(method, 0) + 1
            
            if col.get('characteristics', {}).get('is_ordinal', False):
                ordinal_count += 1
            if col.get('characteristics', {}).get('is_high_cardinality', False):
                high_cardinality_count += 1
    
    overall_recommendation = {
        "total_columns": total_columns,
        "columns_needing_encoding": columns_needing_encoding,
        "encoding_percentage": round((columns_needing_encoding / total_columns) * 100, 2) if total_columns > 0 else 0,
        "method_distribution": method_counts,
        "ordinal_columns": ordinal_count,
        "high_cardinality_columns": high_cardinality_count,
        "overall_need": columns_needing_encoding > 0  # Always need encoding for categorical data
    }
    
    return overall_recommendation

def create_encoding_visualizations(df: pd.DataFrame, column_analysis: Dict) -> Dict[str, Any]:
    """Create visualizations for encoding analysis"""
    
    visualizations = {}
    
    # Select categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    if len(categorical_columns) == 0:
        return {"error": "No categorical columns for visualization"}
    
    # 1. Cardinality comparison
    cardinality_data = []
    for col in categorical_columns:
        if col in column_analysis and 'statistics' in column_analysis[col]:
            stats = column_analysis[col]['statistics']
            cardinality_data.append({
                'column': col,
                'unique_count': stats['unique_count'],
                'cardinality_ratio': stats['cardinality_ratio']
            })
    
    if cardinality_data:
        # Unique count comparison
        unique_count_fig = go.Figure(data=go.Bar(
            x=[d['column'] for d in cardinality_data],
            y=[d['unique_count'] for d in cardinality_data],
            name="Unique Count"
        ))
        unique_count_fig.update_layout(
            title="Unique Values Count by Column",
            xaxis_title="Columns",
            yaxis_title="Unique Count",
            height=400
        )
        visualizations["unique_count_comparison"] = unique_count_fig.to_dict()
        
        # Cardinality ratio comparison
        cardinality_ratio_fig = go.Figure(data=go.Bar(
            x=[d['column'] for d in cardinality_data],
            y=[d['cardinality_ratio'] for d in cardinality_data],
            name="Cardinality Ratio"
        ))
        cardinality_ratio_fig.update_layout(
            title="Cardinality Ratio by Column",
            xaxis_title="Columns",
            yaxis_title="Cardinality Ratio",
            height=400
        )
        visualizations["cardinality_ratio_comparison"] = cardinality_ratio_fig.to_dict()
    
    # 2. Encoding method distribution
    methods = []
    ordinal_flags = []
    
    for col in categorical_columns:
        if col in column_analysis and 'encoding_assessment' in column_analysis[col]:
            assessment = column_analysis[col]['encoding_assessment']
            methods.append(assessment.get('recommended_method', 'None'))
            ordinal_flags.append(column_analysis[col].get('characteristics', {}).get('is_ordinal', False))
    
    if methods:
        # Method distribution
        method_counts = {}
        for method in methods:
            method_counts[method] = method_counts.get(method, 0) + 1
        
        method_fig = go.Figure(data=go.Bar(
            x=list(method_counts.keys()),
            y=list(method_counts.values()),
            name="Recommended Method"
        ))
        method_fig.update_layout(
            title="Recommended Encoding Methods",
            xaxis_title="Method",
            yaxis_title="Count",
            height=400
        )
        visualizations["method_distribution"] = method_fig.to_dict()
        
        # Ordinal vs Nominal
        ordinal_count = sum(ordinal_flags)
        nominal_count = len(ordinal_flags) - ordinal_count
        
        ordinal_pie_fig = go.Figure(data=go.Pie(
            labels=['Ordinal', 'Nominal'],
            values=[ordinal_count, nominal_count],
            hole=0.3
        ))
        ordinal_pie_fig.update_layout(
            title="Ordinal vs Nominal Categories",
            height=400
        )
        visualizations["ordinal_vs_nominal"] = ordinal_pie_fig.to_dict()
    
    return visualizations

def apply_encoding(df: pd.DataFrame, method: str = "Label Encoding", columns: List[str] = None) -> Dict[str, Any]:
    """
    Apply encoding to the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    method : str
        Encoding method: "Label Encoding", "One-Hot Encoding", "Target Encoding"
    columns : List[str]
        Specific columns to encode (if None, encode all categorical columns)
    
    Returns:
    --------
    Dict[str, Any]
        Results of encoding
    """
    
    try:
        from sklearn.preprocessing import LabelEncoder
        import pandas as pd
        
        categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        columns_to_encode = columns or categorical_columns
        
        df_encoded = df.copy()
        encoders = {}
        
        if method == "Label Encoding":
            for column in columns_to_encode:
                if column in categorical_columns:
                    le = LabelEncoder()
                    df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
                    encoders[column] = {
                        "type": "LabelEncoder",
                        "classes": le.classes_.tolist()
                    }
                    
        elif method == "One-Hot Encoding":
            for column in columns_to_encode:
                if column in categorical_columns:
                    # Create dummy variables
                    dummies = pd.get_dummies(df_encoded[column], prefix=column)
                    # Drop original column and add dummies
                    df_encoded = df_encoded.drop(columns=[column])
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                    encoders[column] = {
                        "type": "OneHotEncoder",
                        "dummy_columns": dummies.columns.tolist()
                    }
                    
        elif method == "Target Encoding":
            # This would require target variable - simplified version
            for column in columns_to_encode:
                if column in categorical_columns:
                    # Simple frequency encoding as fallback
                    value_counts = df_encoded[column].value_counts(normalize=True)
                    df_encoded[column] = df_encoded[column].map(value_counts)
                    encoders[column] = {
                        "type": "TargetEncoder",
                        "encoding_values": value_counts.to_dict()
                    }
        
        return {
            "encoded_dataframe": df_encoded,
            "method": method,
            "columns_encoded": columns_to_encode,
            "encoders": encoders,
            "original_shape": df.shape,
            "encoded_shape": df_encoded.shape
        }
        
    except ImportError:
        return {
            "error": "scikit-learn not available for encoding"
        }
    except Exception as e:
        return {
            "error": f"Error in encoding: {str(e)}"
        }
