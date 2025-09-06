"""Intelligent data type and format conversion using AI."""
import pandas as pd
import numpy as np
import os
import sys

# Add the project root to the path so we can import from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.ai_utils import generate_insight

def run(df, params=None):
    """
    Analyze and intelligently convert data types using AI assistance
    
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
        - original_dtypes: Original data types
        - converted_dtypes: New data types after conversion
        - conversion_details: Details of conversions made
        - ai_insights: AI-generated insights about the conversions
    """
    # Store original data types
    original_dtypes = df.dtypes.apply(str).to_dict()
    
    # Create a copy of the DataFrame for conversion
    df_converted = df.copy()
    
    # Track conversion details
    conversion_details = {}
    
    # Analyze each column
    for column in df.columns:
        # Skip columns that are already numeric or datetime
        if pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_datetime64_dtype(df[column]):
            continue
            
        # Only process string/object columns
        if df[column].dtype == 'object':
            # Sample non-null values for type inference
            sample_values = df[column].dropna().sample(min(10, len(df[column].dropna()))).tolist()
            if not sample_values:
                continue
                
            # Initialize conversion details
            conversion_details[column] = {
                'original_type': str(df[column].dtype),
                'sample_values': sample_values[:5],  # Show up to 5 sample values
                'conversion_applied': None,
                'new_type': None
            }
            
            # Try numeric conversion with various formats
            try:
                # Handle various number formats: "1,888", "1.888,00", "$1,888", etc.
                # First, try to clean the strings
                cleaned_values = df[column].astype(str).str.replace('$', '')
                cleaned_values = cleaned_values.str.replace('%', '')
                
                # Try European format (1.888,00) -> 1888.00
                # Check for values containing both separators
                separator_count = cleaned_values.str.count('[,.]').fillna(0)
                multiple_separators = (separator_count >= 2)
                
                if multiple_separators.any():
                    # Check if it's likely European format
                    european_format = False
                    for val in sample_values:
                        if isinstance(val, str):
                            comma_idx = val.rfind(',')
                            dot_idx = val.rfind('.')
                            if comma_idx != -1 and dot_idx != -1 and comma_idx > dot_idx:
                                european_format = True
                                break
                    
                    if european_format:
                        # Convert European format to US format
                        temp_values = cleaned_values.str.replace('.', '')
                        temp_values = temp_values.str.replace(',', '.')
                        numeric_vals = pd.to_numeric(temp_values, errors='coerce')
                        if numeric_vals.notna().mean() > 0.8:  # If more than 80% are valid numbers
                            df_converted[column] = numeric_vals
                            conversion_details[column]['conversion_applied'] = 'European numeric format'
                            conversion_details[column]['new_type'] = str(df_converted[column].dtype)
                            continue
                
                # Standard US format (1,888.00) -> 1888.00
                temp_values = cleaned_values.str.replace(',', '')
                numeric_vals = pd.to_numeric(temp_values, errors='coerce')
                if numeric_vals.notna().mean() > 0.8:  # If more than 80% are valid numbers
                    df_converted[column] = numeric_vals
                    conversion_details[column]['conversion_applied'] = 'US numeric format'
                    conversion_details[column]['new_type'] = str(df_converted[column].dtype)
                    continue
            except (ValueError, AttributeError):
                pass
            
            # Try datetime conversion with multiple formats
            try:
                df_converted[column] = pd.to_datetime(df[column], errors='coerce')
                # Only keep the conversion if most values were successfully converted
                if df_converted[column].notna().mean() > 0.5:  # More than 50% converted successfully
                    conversion_details[column]['conversion_applied'] = 'datetime'
                    conversion_details[column]['new_type'] = str(df_converted[column].dtype)
                    continue
                else:
                    # Revert back if conversion wasn't successful for most values
                    df_converted[column] = df[column].copy()
            except (ValueError, TypeError):
                pass
            
            # Try boolean conversion for columns with few unique values
            if df[column].nunique() <= 2:
                try:
                    # Create a more comprehensive boolean mapping
                    bool_map = {
                        'true': True, 'false': False,
                        'yes': True, 'no': False,
                        'y': True, 'n': False,
                        't': True, 'f': False,
                        '1': True, '0': False,
                        1: True, 0: False,
                        'on': True, 'off': False,
                        'enable': True, 'disable': False,
                        'enabled': True, 'disabled': False
                    }
                    
                    # Case-insensitive mapping
                    lower_values = df[column].astype(str).str.lower()
                    if all(val in bool_map for val in lower_values.unique() if not pd.isna(val)):
                        df_converted[column] = lower_values.map(bool_map)
                        conversion_details[column]['conversion_applied'] = 'boolean'
                        conversion_details[column]['new_type'] = str(df_converted[column].dtype)
                        continue
                except (ValueError, TypeError):
                    pass
            
            # If no conversion was applied
            if conversion_details[column]['conversion_applied'] is None:
                conversion_details[column]['conversion_applied'] = 'none'
                conversion_details[column]['new_type'] = str(df[column].dtype)
    
    # Get new data types
    converted_dtypes = df_converted.dtypes.apply(str).to_dict()
    
    # Generate AI insights about the conversions
    conversion_summary = {
        'total_columns': len(df.columns),
        'converted_columns': sum(1 for col in conversion_details if conversion_details[col]['conversion_applied'] != 'none'),
        'conversion_types': {}
    }
    
    # Count conversion types
    for col in conversion_details:
        conv_type = conversion_details[col]['conversion_applied']
        if conv_type != 'none':
            conversion_summary['conversion_types'][conv_type] = conversion_summary['conversion_types'].get(conv_type, 0) + 1
    
    # Generate AI insights
    insight_prompt = f"""
    Analyze the data type conversions performed on this dataset:
    - Total columns: {conversion_summary['total_columns']}
    - Columns converted: {conversion_summary['converted_columns']}
    - Conversion types applied: {conversion_summary['conversion_types']}
    
    For each conversion type, explain:
    1. Why this conversion is beneficial for data analysis
    2. What potential issues might have been resolved
    3. How this improves data quality
    
    Provide a concise summary of the overall impact of these conversions on data quality and analysis potential.
    """
    
    try:
        ai_insights = generate_insight(insight_prompt, context=conversion_details)
    except Exception as e:
        ai_insights = f"Unable to generate AI insights: {str(e)}"
    
    # Return results
    return {
        'original_dtypes': original_dtypes,
        'converted_dtypes': converted_dtypes,
        'conversion_details': conversion_details,
        'ai_insights': ai_insights,
        'converted_df': df_converted
    }