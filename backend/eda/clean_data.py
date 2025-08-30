import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
import sqlite3
import os
import uuid

def run(df: pd.DataFrame, treatment_options: dict = None) -> dict:
    """
    Clean the dataset by handling missing values based on treatment options
    
    Parameters:
    -----------
    df : pandas DataFrame
        The original dataset to clean
    treatment_options : dict
        Dictionary containing treatment options for each column
        If None, will use default treatment methods based on data analysis
    
    Returns:
    --------
    dict
        Dictionary containing the cleaned dataframe and summary of changes
    """
    if df is None or df.empty:
        return {
            "success": False,
            "message": "No data provided for cleaning",
            "cleaned_df": None,
            "file_id": None
        }
    
    # Make a copy of the original dataframe to avoid modifying it
    cleaned_df = df.copy()
    
    # If no treatment options provided, generate default ones
    if treatment_options is None:
        treatment_options = generate_default_treatment_options(df)
    
    # Track changes made to the dataset
    changes = []
    
    # Apply treatments column by column
    for column, options in treatment_options.items():
        if column not in df.columns:
            continue
            
        method = options.get('method', 'auto')
        
        # Skip columns marked for no treatment
        if method == 'none' or method == 'skip':
            changes.append(f"Column '{column}': No treatment applied (skipped)")
            continue
            
        # Drop columns marked for dropping
        if method == 'drop_column':
            cleaned_df = cleaned_df.drop(columns=[column])
            changes.append(f"Column '{column}': Dropped entirely")
            continue
            
        # Handle missing values based on method
        missing_count = df[column].isna().sum()
        if missing_count == 0:
            changes.append(f"Column '{column}': No missing values")
            continue
            
        # Apply the specified treatment method
        if method == 'drop_rows':
            # Drop rows with missing values in this column
            rows_before = len(cleaned_df)
            cleaned_df = cleaned_df.dropna(subset=[column])
            rows_dropped = rows_before - len(cleaned_df)
            changes.append(f"Column '{column}': Dropped {rows_dropped} rows with missing values")
            
        elif method == 'mean':
            # Impute with mean (numeric columns only)
            if pd.api.types.is_numeric_dtype(df[column]):
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mean())
                changes.append(f"Column '{column}': Imputed {missing_count} missing values with mean")
            else:
                changes.append(f"Column '{column}': Cannot apply mean imputation to non-numeric column")
                
        elif method == 'median':
            # Impute with median (numeric columns only)
            if pd.api.types.is_numeric_dtype(df[column]):
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
                changes.append(f"Column '{column}': Imputed {missing_count} missing values with median")
            else:
                changes.append(f"Column '{column}': Cannot apply median imputation to non-numeric column")
                
        elif method == 'mode':
            # Impute with mode (works for both numeric and categorical)
            mode_value = cleaned_df[column].mode()[0]
            cleaned_df[column] = cleaned_df[column].fillna(mode_value)
            changes.append(f"Column '{column}': Imputed {missing_count} missing values with mode")
            
        elif method == 'constant':
            # Impute with a constant value
            fill_value = options.get('fill_value', 0 if pd.api.types.is_numeric_dtype(df[column]) else 'MISSING')
            cleaned_df[column] = cleaned_df[column].fillna(fill_value)
            changes.append(f"Column '{column}': Imputed {missing_count} missing values with constant '{fill_value}'")
            
        elif method == 'knn':
            # KNN imputation (for numeric columns only)
            if all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
                # Only use KNN if all columns are numeric
                imputer = KNNImputer(n_neighbors=5)
                cleaned_df[column] = imputer.fit_transform(cleaned_df[[column]])[:, 0]
                changes.append(f"Column '{column}': Imputed {missing_count} missing values with KNN")
            else:
                # Fallback to median for mixed data types
                if pd.api.types.is_numeric_dtype(df[column]):
                    cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
                    changes.append(f"Column '{column}': Imputed {missing_count} missing values with median (KNN fallback)")
                else:
                    mode_value = cleaned_df[column].mode()[0]
                    cleaned_df[column] = cleaned_df[column].fillna(mode_value)
                    changes.append(f"Column '{column}': Imputed {missing_count} missing values with mode (KNN fallback)")
        
        elif method == 'auto':
            # Automatic method selection based on data type and distribution
            if pd.api.types.is_numeric_dtype(df[column]):
                # For numeric columns, use median (more robust than mean)
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
                changes.append(f"Column '{column}': Imputed {missing_count} missing values with median (auto)")
            else:
                # For categorical columns, use mode
                mode_value = cleaned_df[column].mode()[0]
                cleaned_df[column] = cleaned_df[column].fillna(mode_value)
                changes.append(f"Column '{column}': Imputed {missing_count} missing values with mode (auto)")
    
    # Save the cleaned dataframe to SQLite database
    file_id = save_to_database(cleaned_df)
    
    # Return the cleaned dataframe and summary
    return {
        "success": True,
        "message": "Data cleaning completed successfully",
        "changes": changes,
        "cleaned_df": cleaned_df,
        "file_id": file_id,
        "stats": {
            "original_rows": len(df),
            "cleaned_rows": len(cleaned_df),
            "original_missing": df.isna().sum().sum(),
            "cleaned_missing": cleaned_df.isna().sum().sum()
        }
    }

def generate_default_treatment_options(df):
    """
    Generate default treatment options based on data analysis
    """
    options = {}
    
    for column in df.columns:
        missing_count = df[column].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        if missing_count == 0:
            # No missing values, no treatment needed
            options[column] = {"method": "none"}
            continue
            
        # Decide treatment method based on missing percentage and data type
        if missing_pct > 30:
            # Too many missing values, recommend dropping the column
            options[column] = {"method": "drop_column"}
        elif pd.api.types.is_numeric_dtype(df[column]):
            # For numeric columns with moderate missing values
            if missing_pct > 5:
                # Use median for numeric columns (more robust than mean)
                options[column] = {"method": "median"}
            else:
                # For small percentages, dropping rows is usually safe
                options[column] = {"method": "drop_rows"}
        else:
            # For categorical columns
            if missing_pct > 5:
                # Use mode for categorical columns
                options[column] = {"method": "mode"}
            else:
                # For small percentages, dropping rows is usually safe
                options[column] = {"method": "drop_rows"}
    
    return options

def save_to_database(df, db_path=None):
    """
    Save the cleaned dataframe to SQLite database
    
    Returns:
    --------
    str
        The file ID for the saved dataframe
    """
    # Generate a unique ID for this cleaned dataset
    file_id = str(uuid.uuid4())
    
    # Determine database path
    if db_path is None:
        # Use default path in the project directory
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'eda.db')
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    
    # Save the dataframe to a table named with the file_id
    table_name = f"cleaned_{file_id.replace('-', '_')}"
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    
    # Save metadata about this cleaned dataset
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS cleaned_datasets (" +
        "id TEXT PRIMARY KEY, " +
        "table_name TEXT, " +
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, " +
        "row_count INTEGER, " +
        "column_count INTEGER)"
    )
    
    cursor.execute(
        "INSERT INTO cleaned_datasets (id, table_name, row_count, column_count) VALUES (?, ?, ?, ?)",
        (file_id, table_name, len(df), len(df.columns))
    )
    
    # Commit and close
    conn.commit()
    conn.close()
    
    return file_id