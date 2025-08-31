import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from .utils import export_plot_as_image

def run(df: pd.DataFrame) -> dict:
    """Analyze missing values in the dataset and suggest treatment methods"""
    # Calculate missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Missing %': missing_pct.values
    })
    
    # Only keep columns with missing values
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    
    if missing_df.empty:
        return {
            "text": "No missing values found in the dataset!",
            "plot": None,
            "treatment": {
                "required": False,
                "message": "No treatment needed as there are no missing values."
            }
        }
    
    # Create missing values visualization
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=['Missing Values Count', 'Missing Values Percentage'],
                       vertical_spacing=0.2,
                       row_heights=[0.5, 0.5])
    
    # Add bar chart for missing counts
    fig.add_trace(
        go.Bar(
            x=missing_df['Column'],
            y=missing_df['Missing Count'],
            name='Missing Count',
            marker_color='#9B177E'
        ),
        row=1, col=1
    )
    
    # Add bar chart for missing percentages
    fig.add_trace(
        go.Bar(
            x=missing_df['Column'],
            y=missing_df['Missing %'],
            name='Missing %',
            marker_color='#E8988A'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        width=800,
        title_text="Missing Values Analysis",
        showlegend=False
    )
    
    # Convert plot to base64 string using utility function
    
    # Analyze patterns and suggest treatment methods
    treatment_suggestions = {}
    for column in missing_df['Column']:
        # Skip non-numeric columns for some analyses
        if pd.api.types.is_numeric_dtype(df[column]):
            # Check if missing values are random (MCAR test)
            # For simplicity, we'll use a basic approach to check if values are missing completely at random
            # by comparing means of data with and without missing values in another column
            
            # Get a reference column (first numeric column that's not the current one)
            ref_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != column]
            if ref_cols:
                ref_col = ref_cols[0]
                # Compare means
                mean_with_missing = df[df[column].isna()][ref_col].mean()
                mean_without_missing = df[~df[column].isna()][ref_col].mean()
                
                # If means are similar, likely MCAR (Missing Completely At Random)
                is_mcar = abs(mean_with_missing - mean_without_missing) < 0.1 * df[ref_col].std()
            else:
                is_mcar = True  # Default if we can't test
            
            # Calculate missing percentage
            missing_pct_val = missing_df.loc[missing_df['Column'] == column, 'Missing %'].values[0]
            
            # Determine treatment method
            if missing_pct_val > 30:
                method = "Consider dropping this column due to excessive missing values"
            elif missing_pct_val > 5:
                if is_mcar:
                    method = "Impute with mean/median (for numeric) or mode (for categorical)"
                else:
                    method = "Consider more advanced imputation methods like KNN or regression imputation"
            else:
                if is_mcar:
                    method = "Safe to drop rows with missing values or use simple imputation"
                else:
                    method = "Impute with mean/median/mode"
        else:
            # For non-numeric columns
            missing_pct_val = missing_df.loc[missing_df['Column'] == column, 'Missing %'].values[0]
            if missing_pct_val > 30:
                method = "Consider dropping this column due to excessive missing values"
            elif missing_pct_val > 5:
                method = "Impute with mode or create a 'Missing' category"
            else:
                method = "Safe to drop rows or impute with mode"
        
        treatment_suggestions[column] = {
            "missing_percent": float(missing_pct_val),
            "treatment_method": method,
            "is_random_missing": bool(is_mcar) if pd.api.types.is_numeric_dtype(df[column]) else None
        }
    
    # Overall treatment recommendation
    overall_missing_pct = missing.sum() / (len(df) * len(df.columns)) * 100
    needs_treatment = overall_missing_pct > 0
    
    treatment_info = {
        "required": needs_treatment,
        "overall_missing_percent": float(overall_missing_pct),
        "column_treatments": treatment_suggestions,
        "message": generate_treatment_message(overall_missing_pct, missing_df)
    }
    
    return {
        "text": "Here's the missing values analysis:",
        "plot": export_plot_as_image(fig, width=800, height=600),
        "data": missing_df.to_dict(),
        "treatment": treatment_info
    }

def generate_treatment_message(overall_missing_pct, missing_df):
    """Generate a human-readable message about missing value treatment"""
    if overall_missing_pct == 0:
        return "No missing values detected. No treatment needed."
    
    if overall_missing_pct < 1:
        return "Very few missing values detected. You can safely drop rows with missing values or use simple imputation methods."
    
    if overall_missing_pct > 30:
        return "High percentage of missing values detected. Consider if this dataset is suitable for your analysis or look into advanced imputation techniques."
    
    # Count columns with different levels of missing values
    high_missing = sum(missing_df['Missing %'] > 20)
    moderate_missing = sum((missing_df['Missing %'] <= 20) & (missing_df['Missing %'] > 5))
    low_missing = sum(missing_df['Missing %'] <= 5)
    
    message = f"Dataset contains {len(missing_df)} columns with missing values. "
    if high_missing > 0:
        message += f"{high_missing} columns have high missing rates (>20%). "
    if moderate_missing > 0:
        message += f"{moderate_missing} columns have moderate missing rates (5-20%). "
    if low_missing > 0:
        message += f"{low_missing} columns have low missing rates (<5%). "
    
    message += "\n\nRecommended approach: "
    if high_missing > 0:
        message += "Consider dropping columns with high missing rates. "
    if moderate_missing > 0 or low_missing > 0:
        message += "For remaining columns, use appropriate imputation methods based on data type and distribution."
    
    return message
