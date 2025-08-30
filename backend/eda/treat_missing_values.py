import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
import json

def run(df: pd.DataFrame, treatment_config: dict = None) -> dict:
    """
    Treat missing values in the dataset based on the provided configuration
    
    Parameters:
    -----------
    df : pandas DataFrame
        The original dataset to clean
    treatment_config : dict
        Dictionary containing treatment options for each column
        If None, will use default treatment methods based on data analysis
    
    Returns:
    --------
    dict
        Dictionary containing the treated dataframe and summary of changes
    """
    if df is None or df.empty:
        return {
            "success": False,
            "text": "No data provided for treatment",
            "plot": None
        }
    
    # Make a copy of the original dataframe to avoid modifying it
    treated_df = df.copy()
    
    # If no treatment config provided, generate default one
    if treatment_config is None:
        treatment_config = generate_default_treatment_config(df)
    
    # Track changes made to the dataset
    changes = []
    columns_treated = []
    
    # Apply treatments column by column
    for column, options in treatment_config.items():
        if column not in df.columns:
            continue
            
        method = options.get('method', 'auto')
        
        # Skip columns marked for no treatment
        if method == 'none' or method == 'skip':
            changes.append(f"Column '{column}': No treatment applied (skipped)")
            continue
            
        # Drop columns marked for dropping
        if method == 'drop_column':
            treated_df = treated_df.drop(columns=[column])
            changes.append(f"Column '{column}': Dropped entirely")
            columns_treated.append({
                "column": column,
                "method": "drop_column",
                "missing_count": int(df[column].isna().sum()),
                "missing_percent": float((df[column].isna().sum() / len(df)) * 100)
            })
            continue
            
        # Handle missing values based on method
        missing_count = df[column].isna().sum()
        if missing_count == 0:
            changes.append(f"Column '{column}': No missing values")
            continue
            
        missing_percent = (missing_count / len(df)) * 100
        
        # Apply the specified treatment method
        if method == 'drop_rows':
            # Drop rows with missing values in this column
            rows_before = len(treated_df)
            treated_df = treated_df.dropna(subset=[column])
            rows_dropped = rows_before - len(treated_df)
            changes.append(f"Column '{column}': Dropped {rows_dropped} rows with missing values")
            columns_treated.append({
                "column": column,
                "method": "drop_rows",
                "rows_dropped": int(rows_dropped),
                "missing_count": int(missing_count),
                "missing_percent": float(missing_percent)
            })
            
        elif method == 'mean':
            # Impute with mean (numeric columns only)
            if pd.api.types.is_numeric_dtype(df[column]):
                mean_value = df[column].mean()
                treated_df[column] = treated_df[column].fillna(mean_value)
                changes.append(f"Column '{column}': Imputed {missing_count} missing values with mean ({mean_value:.2f})")
                columns_treated.append({
                    "column": column,
                    "method": "mean",
                    "imputed_value": float(mean_value),
                    "missing_count": int(missing_count),
                    "missing_percent": float(missing_percent)
                })
            else:
                changes.append(f"Column '{column}': Cannot apply mean imputation to non-numeric column")
                
        elif method == 'median':
            # Impute with median (numeric columns only)
            if pd.api.types.is_numeric_dtype(df[column]):
                median_value = df[column].median()
                treated_df[column] = treated_df[column].fillna(median_value)
                changes.append(f"Column '{column}': Imputed {missing_count} missing values with median ({median_value:.2f})")
                columns_treated.append({
                    "column": column,
                    "method": "median",
                    "imputed_value": float(median_value),
                    "missing_count": int(missing_count),
                    "missing_percent": float(missing_percent)
                })
            else:
                changes.append(f"Column '{column}': Cannot apply median imputation to non-numeric column")
                
        elif method == 'mode':
            # Impute with mode (works for both numeric and categorical)
            mode_value = df[column].mode()[0]
            treated_df[column] = treated_df[column].fillna(mode_value)
            mode_display = str(mode_value)
            if pd.api.types.is_numeric_dtype(df[column]):
                mode_display = f"{mode_value:.2f}"
            changes.append(f"Column '{column}': Imputed {missing_count} missing values with mode ({mode_display})")
            columns_treated.append({
                "column": column,
                "method": "mode",
                "imputed_value": mode_value if pd.api.types.is_numeric_dtype(df[column]) else str(mode_value),
                "missing_count": int(missing_count),
                "missing_percent": float(missing_percent)
            })
            
        elif method == 'constant':
            # Impute with a constant value
            fill_value = options.get('fill_value', 0 if pd.api.types.is_numeric_dtype(df[column]) else 'MISSING')
            treated_df[column] = treated_df[column].fillna(fill_value)
            changes.append(f"Column '{column}': Imputed {missing_count} missing values with constant '{fill_value}'")
            columns_treated.append({
                "column": column,
                "method": "constant",
                "imputed_value": float(fill_value) if pd.api.types.is_numeric_dtype(df[column]) else str(fill_value),
                "missing_count": int(missing_count),
                "missing_percent": float(missing_percent)
            })
            
        elif method == 'knn':
            # KNN imputation (for numeric columns only)
            if pd.api.types.is_numeric_dtype(df[column]):
                # Get only numeric columns for KNN
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 1:  # Need at least 2 columns for KNN
                    # Create a temporary dataframe with only numeric columns
                    temp_df = df[numeric_cols].copy()
                    # Fill any missing values in other columns with median (to make KNN work)
                    for col in numeric_cols:
                        if col != column and temp_df[col].isna().sum() > 0:
                            temp_df[col] = temp_df[col].fillna(temp_df[col].median())
                    
                    # Apply KNN imputation
                    imputer = KNNImputer(n_neighbors=min(5, len(df)-1))
                    imputed_values = imputer.fit_transform(temp_df)
                    # Update only the target column
                    col_idx = numeric_cols.index(column)
                    treated_df[column] = imputed_values[:, col_idx]
                    changes.append(f"Column '{column}': Imputed {missing_count} missing values with KNN")
                    columns_treated.append({
                        "column": column,
                        "method": "knn",
                        "missing_count": int(missing_count),
                        "missing_percent": float(missing_percent)
                    })
                else:
                    # Fallback to median if not enough numeric columns
                    median_value = df[column].median()
                    treated_df[column] = treated_df[column].fillna(median_value)
                    changes.append(f"Column '{column}': Imputed {missing_count} missing values with median (KNN fallback)")
                    columns_treated.append({
                        "column": column,
                        "method": "median",
                        "imputed_value": float(median_value),
                        "missing_count": int(missing_count),
                        "missing_percent": float(missing_percent)
                    })
            else:
                # Fallback to mode for non-numeric columns
                mode_value = df[column].mode()[0]
                treated_df[column] = treated_df[column].fillna(mode_value)
                changes.append(f"Column '{column}': Imputed {missing_count} missing values with mode (KNN fallback)")
                columns_treated.append({
                    "column": column,
                    "method": "mode",
                    "imputed_value": str(mode_value),
                    "missing_count": int(missing_count),
                    "missing_percent": float(missing_percent)
                })
        
        elif method == 'auto':
            # Automatic method selection based on data type and distribution
            if pd.api.types.is_numeric_dtype(df[column]):
                # For numeric columns, use median (more robust than mean)
                median_value = df[column].median()
                treated_df[column] = treated_df[column].fillna(median_value)
                changes.append(f"Column '{column}': Imputed {missing_count} missing values with median (auto)")
                columns_treated.append({
                    "column": column,
                    "method": "median",
                    "imputed_value": float(median_value),
                    "missing_count": int(missing_count),
                    "missing_percent": float(missing_percent)
                })
            else:
                # For categorical columns, use mode
                mode_value = df[column].mode()[0]
                treated_df[column] = treated_df[column].fillna(mode_value)
                changes.append(f"Column '{column}': Imputed {missing_count} missing values with mode (auto)")
                columns_treated.append({
                    "column": column,
                    "method": "mode",
                    "imputed_value": str(mode_value),
                    "missing_count": int(missing_count),
                    "missing_percent": float(missing_percent)
                })
    
    # Generate summary text
    summary_text = generate_summary_text(changes, df, treated_df)
    
    # Return the treated dataframe and summary
    return {
        "success": True,
        "text": summary_text,
        "treated_df": treated_df,
        "changes": changes,
        "columns_treated": columns_treated,
        "stats": {
            "original_rows": len(df),
            "treated_rows": len(treated_df),
            "original_missing": int(df.isna().sum().sum()),
            "treated_missing": int(treated_df.isna().sum().sum()),
            "treatment_effectiveness": float(100 * (1 - treated_df.isna().sum().sum() / max(1, df.isna().sum().sum())))
        }
    }

def generate_default_treatment_config(df):
    """
    Generate default treatment configuration based on data analysis
    """
    config = {}
    
    for column in df.columns:
        missing_count = df[column].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        if missing_count == 0:
            # No missing values, no treatment needed
            config[column] = {"method": "none"}
            continue
            
        # Decide treatment method based on missing percentage and data type
        if missing_pct > 30:
            # Too many missing values, recommend dropping the column
            config[column] = {"method": "drop_column"}
        elif pd.api.types.is_numeric_dtype(df[column]):
            # For numeric columns with moderate missing values
            if missing_pct > 5:
                # Use median for numeric columns (more robust than mean)
                config[column] = {"method": "median"}
            else:
                # For small percentages, dropping rows is usually safe
                config[column] = {"method": "drop_rows"}
        else:
            # For categorical columns
            if missing_pct > 5:
                # Use mode for categorical columns
                config[column] = {"method": "mode"}
            else:
                # For small percentages, dropping rows is usually safe
                config[column] = {"method": "drop_rows"}
    
    return config

def generate_summary_text(changes, original_df, treated_df):
    """
    Generate a human-readable summary of the changes made
    """
    original_missing = original_df.isna().sum().sum()
    treated_missing = treated_df.isna().sum().sum()
    
    if original_missing == 0:
        return "No missing values found in the dataset. No treatment was needed."
    
    if original_missing == treated_missing:
        return "Treatment did not reduce any missing values. Consider different treatment methods."
    
    # Calculate effectiveness
    effectiveness = 100 * (1 - treated_missing / original_missing)
    
    summary = f"Missing values treatment summary:\n\n"
    summary += f"• Original dataset: {original_df.shape[0]} rows, {original_df.shape[1]} columns, {original_missing} missing values\n"
    summary += f"• Treated dataset: {treated_df.shape[0]} rows, {treated_df.shape[1]} columns, {treated_missing} missing values\n"
    summary += f"• Treatment effectiveness: {effectiveness:.1f}% of missing values resolved\n\n"
    
    # Categorize changes
    dropped_columns = []
    dropped_rows_columns = []
    imputed_columns = []
    
    for change in changes:
        if "Dropped entirely" in change:
            dropped_columns.append(change.split("'")[1])
        elif "Dropped" in change and "rows" in change:
            dropped_rows_columns.append(change.split("'")[1])
        elif "Imputed" in change:
            imputed_columns.append(change.split("'")[1])
    
    # Add details about changes
    if dropped_columns:
        summary += f"• Dropped {len(dropped_columns)} columns with excessive missing values: {', '.join(dropped_columns)}\n"
    
    if dropped_rows_columns:
        summary += f"• Dropped rows with missing values in {len(dropped_rows_columns)} columns: {', '.join(dropped_rows_columns)}\n"
    
    if imputed_columns:
        summary += f"• Imputed missing values in {len(imputed_columns)} columns: {', '.join(imputed_columns)}\n"
    
    return summary