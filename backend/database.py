import sqlite3
import os
import pandas as pd
import uuid
from datetime import datetime

import json
from typing import Optional, Dict, Any

class Database:
    def __init__(self, db_path=None):
        """
        Initialize the database connection
        
        Parameters:
        -----------
        db_path : str, optional
            Path to the SQLite database file. If None, a default path will be used.
        """
        if db_path is None:
            # Use default path in the project directory
            self.db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'eda.db')
        else:
            self.db_path = db_path
            
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize the database
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Initialize the database schema
        self._init_db()
    
    def _init_db(self):
        """
        Initialize the database with required tables if they don't exist
        """
        try:
            self.conn.execute("BEGIN")
            self._create_tables()
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e
    
    def _create_tables(self):
        """Create all necessary database tables."""
        # Create table for uploaded files metadata
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS uploaded_files (
            id TEXT PRIMARY KEY,
            original_filename TEXT,
            table_name TEXT,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_size INTEGER,
            row_count INTEGER,
            column_count INTEGER,
            status TEXT
        )
        """)
        
        # Create table for file summaries
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS file_summaries (
            file_id TEXT PRIMARY KEY,
            summary_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (file_id) REFERENCES uploaded_files (id)
        )
        """)
        
        # Create table for cleaning steps
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS cleaning_steps (
            id TEXT PRIMARY KEY,
            file_id TEXT,
            step_type TEXT,
            step_details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (file_id) REFERENCES uploaded_files (id)
        )
        """)
    
    def save_file_summary(self, file_id: str, summary: Dict[str, Any]) -> None:
        """
        Save or update file summary data
        
        Parameters:
        -----------
        file_id : str
            The ID of the file
        summary : dict
            Summary data to save
        """
        try:
            self.conn.execute("BEGIN")
            self.cursor.execute("""
            INSERT OR REPLACE INTO file_summaries (file_id, summary_data)
            VALUES (?, ?)
            """, (file_id, json.dumps(summary)))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e
    
    def get_file_summary(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve file summary data
        
        Parameters:
        -----------
        file_id : str
            The ID of the file
        
        Returns:
        --------
        dict or None
            The summary data if found, None otherwise
        """
        self.cursor.execute("""
        SELECT summary_data
        FROM file_summaries
        WHERE file_id = ?
        """, (file_id,))
        
        result = self.cursor.fetchone()
        if result:
            return json.loads(result[0])
        return None
    
    def add_cleaning_step(self, file_id: str, step_type: str, step_details: Dict[str, Any]) -> None:
        """
        Add a cleaning step for a file
        
        Parameters:
        -----------
        file_id : str
            The ID of the file
        step_type : str
            Type of cleaning step (e.g., 'missing_values', 'outliers')
        step_details : dict
            Details of the cleaning step
        """
        step_id = str(uuid.uuid4())
        try:
            self.conn.execute("BEGIN")
            self.cursor.execute("""
            INSERT INTO cleaning_steps (id, file_id, step_type, step_details)
            VALUES (?, ?, ?, ?)
            """, (step_id, file_id, step_type, json.dumps(step_details)))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e
    
    def __del__(self):
        """Clean up database connection"""
        try:
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
        except:
            pass
    
    def save_uploaded_file(self, df, original_filename):
        """
        Save an uploaded DataFrame to the database
        
        Parameters:
        -----------
        df : pandas DataFrame
            The DataFrame to save
        original_filename : str
            The original filename of the uploaded file
            
        Returns:
        --------
        str
            The file ID for the saved DataFrame
        """
        # Generate a unique ID for this file
        file_id = str(uuid.uuid4())
        
        # Create a table name based on the file ID
        table_name = f"file_{file_id.replace('-', '_')}"
        
        # Connect to the database
        conn = sqlite3.connect(self.db_path)
        
        # Ensure all data types are compatible with SQLite
        # Convert any problematic types before saving
        for col in df.columns:
            # Convert any complex numpy types to standard Python types
            if pd.api.types.is_integer_dtype(df[col]):
                df[col] = df[col].astype('int64')
            elif pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].astype('float64')
            elif pd.api.types.is_bool_dtype(df[col]):
                df[col] = df[col].astype('bool')
            elif pd.api.types.is_datetime64_dtype(df[col]):
                df[col] = df[col].astype(str)  # Convert datetime to string for SQLite compatibility
        
        # Save the DataFrame to a table
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        # Save metadata about this file
        cursor = conn.cursor()
        
        # Calculate file size from DataFrame memory usage instead of file path
        file_size = df.memory_usage(deep=True).sum()
        
        cursor.execute(
            "INSERT INTO uploaded_files (id, original_filename, table_name, upload_time, file_size, row_count, column_count, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (file_id, original_filename, table_name, datetime.now(), file_size, len(df), len(df.columns), 'ready')
        )
        
        # Commit and close
        conn.commit()
        conn.close()
        
        return file_id
    
    def save_cleaned_dataset(self, df, original_file_id):
        """
        Save a cleaned DataFrame to the database
        
        Parameters:
        -----------
        df : pandas DataFrame
            The cleaned DataFrame to save
        original_file_id : str
            The file ID of the original uploaded file
            
        Returns:
        --------
        str
            The file ID for the saved cleaned DataFrame
        """
        # Generate a unique ID for this cleaned dataset
        file_id = str(uuid.uuid4())
        
        # Create a table name based on the file ID
        table_name = f"cleaned_{file_id.replace('-', '_')}"
        
        # Connect to the database
        conn = sqlite3.connect(self.db_path)
        
        # Save the DataFrame to a table
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        # Save metadata about this cleaned dataset
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO cleaned_datasets (id, original_file_id, table_name, created_at, row_count, column_count) VALUES (?, ?, ?, ?, ?, ?)",
            (file_id, original_file_id, table_name, datetime.now(), len(df), len(df.columns))
        )
        
        # Commit and close
        conn.commit()
        conn.close()
        
        return file_id
    
    def get_dataframe(self, file_id, is_cleaned=False):
        """
        Retrieve a DataFrame from the database
        
        Parameters:
        -----------
        file_id : str
            The file ID of the DataFrame to retrieve
        is_cleaned : bool, optional
            Whether to retrieve a cleaned dataset or an original uploaded file
            
        Returns:
        --------
        pandas DataFrame or None
            The retrieved DataFrame, or None if not found
        """
        # Connect to the database
        conn = sqlite3.connect(self.db_path)
        
        # Get the table name for this file ID
        cursor = conn.cursor()
        if is_cleaned:
            cursor.execute("SELECT table_name FROM cleaned_datasets WHERE id = ?", (file_id,))
        else:
            cursor.execute("SELECT table_name FROM uploaded_files WHERE id = ?", (file_id,))
            
        result = cursor.fetchone()
        if result is None:
            conn.close()
            return None
            
        table_name = result[0]
        
        # Load the DataFrame from the table
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        
        # Close the connection
        conn.close()
        
        return df
    
    def get_file_status(self, file_id):
        """
        Get the status of an uploaded file
        
        Parameters:
        -----------
        file_id : str
            The file ID to check
            
        Returns:
        --------
        dict
            A dictionary containing the file status and metadata
        """
        # Connect to the database
        conn = sqlite3.connect(self.db_path)
        
        # Get the file metadata
        cursor = conn.cursor()
        cursor.execute("""
        SELECT id, original_filename, upload_time, file_size, row_count, column_count, status 
        FROM uploaded_files 
        WHERE id = ?
        """, (file_id,))
        
        result = cursor.fetchone()
        if result is None:
            conn.close()
            return {"status": "not_found", "message": "File not found"}
            
        # Convert to dictionary
        file_info = {
            "id": result[0],
            "original_filename": result[1],
            "upload_time": result[2],
            "file_size": result[3],
            "row_count": result[4],
            "column_count": result[5],
            "status": result[6]
        }
        
        # Close the connection
        conn.close()
        
        return file_info
    
    def update_file_status(self, file_id, status):
        """
        Update the status of an uploaded file
        
        Parameters:
        -----------
        file_id : str
            The file ID to update
        status : str
            The new status value
            
        Returns:
        --------
        bool
            True if the update was successful, False otherwise
        """
        # Connect to the database
        conn = sqlite3.connect(self.db_path)
        
        # Update the status
        cursor = conn.cursor()
        cursor.execute("UPDATE uploaded_files SET status = ? WHERE id = ?", (status, file_id))
        
        # Check if the update was successful
        success = cursor.rowcount > 0
        
        # Commit and close
        conn.commit()
        conn.close()
        
        return success
    
    def list_uploaded_files(self):
        """
        List all uploaded files
        
        Returns:
        --------
        list
            A list of dictionaries containing file metadata
        """
        # Connect to the database
        conn = sqlite3.connect(self.db_path)
        
        # Get all file metadata
        cursor = conn.cursor()
        cursor.execute("""
        SELECT id, original_filename, upload_time, file_size, row_count, column_count, status 
        FROM uploaded_files 
        ORDER BY upload_time DESC
        """)
        
        results = cursor.fetchall()
        
        # Convert to list of dictionaries
        files = []
        for result in results:
            files.append({
                "id": result[0],
                "original_filename": result[1],
                "upload_time": result[2],
                "file_size": result[3],
                "row_count": result[4],
                "column_count": result[5],
                "status": result[6]
            })
        
        # Close the connection
        conn.close()
        
        return files
    
    def list_cleaned_datasets(self, original_file_id=None):
        """
        List all cleaned datasets, optionally filtered by original file ID
        
        Parameters:
        -----------
        original_file_id : str, optional
            The file ID of the original uploaded file to filter by
            
        Returns:
        --------
        list
            A list of dictionaries containing cleaned dataset metadata
        """
        # Connect to the database
        conn = sqlite3.connect(self.db_path)
        
        # Get all cleaned dataset metadata
        cursor = conn.cursor()
        if original_file_id is None:
            cursor.execute("""
            SELECT id, original_file_id, created_at, row_count, column_count 
            FROM cleaned_datasets 
            ORDER BY created_at DESC
            """)
        else:
            cursor.execute("""
            SELECT id, original_file_id, created_at, row_count, column_count 
            FROM cleaned_datasets 
            WHERE original_file_id = ? 
            ORDER BY created_at DESC
            """, (original_file_id,))
        
        results = cursor.fetchall()
        
        # Convert to list of dictionaries
        datasets = []
        for result in results:
            datasets.append({
                "id": result[0],
                "original_file_id": result[1],
                "created_at": result[2],
                "row_count": result[3],
                "column_count": result[4]
            })
        
        # Close the connection
        conn.close()
        
        return datasets
    
    def delete_file(self, file_id, delete_cleaned=False):
        """
        Delete a file and its associated table from the database
        
        Parameters:
        -----------
        file_id : str
            The file ID to delete
        delete_cleaned : bool, optional
            Whether to also delete any cleaned datasets derived from this file
            
        Returns:
        --------
        bool
            True if the deletion was successful, False otherwise
        """
        # Connect to the database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Start a transaction
            cursor.execute("BEGIN TRANSACTION")
            
            # Get the table name for this file ID
            cursor.execute("SELECT table_name FROM uploaded_files WHERE id = ?", (file_id,))
            result = cursor.fetchone()
            if result is None:
                # File not found, rollback and return False
                cursor.execute("ROLLBACK")
                conn.close()
                return False
                
            table_name = result[0]
            
            # If requested, delete any cleaned datasets derived from this file
            if delete_cleaned:
                # Get all cleaned datasets derived from this file
                cursor.execute("SELECT id, table_name FROM cleaned_datasets WHERE original_file_id = ?", (file_id,))
                cleaned_datasets = cursor.fetchall()
                
                # Delete each cleaned dataset
                for cleaned_id, cleaned_table in cleaned_datasets:
                    # Drop the table
                    cursor.execute(f"DROP TABLE IF EXISTS {cleaned_table}")
                    
                    # Delete the metadata
                    cursor.execute("DELETE FROM cleaned_datasets WHERE id = ?", (cleaned_id,))
            
            # Drop the table for this file
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            # Delete the metadata for this file
            cursor.execute("DELETE FROM uploaded_files WHERE id = ?", (file_id,))
            
            # Commit the transaction
            cursor.execute("COMMIT")
            
            # Close the connection
            conn.close()
            
            return True
            
        except Exception as e:
            # If an error occurs, rollback the transaction
            cursor.execute("ROLLBACK")
            conn.close()
            print(f"Error deleting file: {e}")
            return False