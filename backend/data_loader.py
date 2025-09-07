"""Module for handling direct CSV to database operations."""
import pandas as pd
from sqlalchemy import create_engine, text
import sqlite3
import os
from typing import Optional, Dict, Any
import uuid

class DataLoader:
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the data loader with database connection."""
        if db_path is None:
            self.db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'db', 'data', 'eda.db')
        else:
            self.db_path = db_path
            print("path=",db_path)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Create SQLAlchemy engine
        self.engine = create_engine(f'sqlite:///{self.db_path}')

    def load_csv_to_db(self, file_path: str, table_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Load CSV file directly into SQLite database using chunks to handle large files.
        
        Args:
            file_path: Path to the CSV file
            table_name: Optional table name, if not provided will generate one
            
        Returns:
            Dictionary containing status and metadata
        """
        try:
            # Generate unique table name if not provided
            if table_name is None:
                table_name = f"dataset_{str(uuid.uuid4()).replace('-', '_')}"
            
            # Read CSV in chunks and write to database
            chunksize = 10000  # Adjust based on your memory constraints
            chunks = pd.read_csv(file_path, chunksize=chunksize)
            
            for i, chunk in enumerate(chunks):
                # Clean column names (remove special characters, spaces)
                chunk.columns = [c.replace(' ', '_').replace('-', '_').replace('[', '').replace(']', '') 
                               for c in chunk.columns]
                
                # Write to database
                if i == 0:
                    # First chunk - create table
                    chunk.to_sql(table_name, self.engine, if_exists='replace', index=False)
                else:
                    # Append subsequent chunks
                    chunk.to_sql(table_name, self.engine, if_exists='append', index=False)

            # Get metadata about the loaded data
            with self.engine.connect() as conn:
                # Get row count
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                row_count = result.scalar()
                
                # Get column information
                result = conn.execute(text(f"PRAGMA table_info({table_name})"))
                columns = [dict(r) for r in result]

            return {
                'status': 'success',
                'table_name': table_name,
                'row_count': row_count,
                'columns': columns,
                'message': f'Data successfully loaded into table {table_name}'
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def get_data_from_table(self, table_name: str, limit: int = 1000) -> pd.DataFrame:
        """
        Retrieve data from a table in chunks.
        
        Args:
            table_name: Name of the table to query
            limit: Maximum number of rows to return
            
        Returns:
            Pandas DataFrame containing the data
        """
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        return pd.read_sql(query, self.engine)

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a table."""
        with self.engine.connect() as conn:
            # Get row count
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            row_count = result.scalar()
            
            # Get column information
            result = conn.execute(text(f"PRAGMA table_info({table_name})"))
            columns = [dict(r) for r in result]

        return {
            'table_name': table_name,
            'row_count': row_count,
            'columns': columns
        }
