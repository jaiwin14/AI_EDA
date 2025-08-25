import pandas as pd
import numpy as np
from typing import Dict, List, Any
from utils.io_utils import DatasetContext, infer_column_types
import logging

logger = logging.getLogger(__name__)

class IngestAgent:
    """Agent responsible for data ingestion and initial schema analysis"""
    
    def __init__(self):
        self.name = "Ingestion Agent"
        
    def process(self, df: pd.DataFrame, filename: str = "") -> DatasetContext:
        """
        Process uploaded dataset and create context with schema information
        
        Args:
            df: Input DataFrame
            filename: Original filename
            
        Returns:
            DatasetContext with schema and warnings
        """
        context = DatasetContext(df, filename)
        
        # Basic dataset info
        logger.info(f"Processing dataset: {filename} with shape {df.shape}")
        
        # Check for memory issues
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        if memory_mb > 500:  # > 500MB
            context.add_warning(f"Large dataset ({memory_mb:.1f}MB). Consider sampling for faster processing.")
            
        # Infer column types and characteristics
        context.schema_info = infer_column_types(df)
        
        # Check for problematic columns
        self._check_data_quality(context)
        
        # Clean obvious issues
        context.df = self._basic_cleanup(context.df, context)
        
        return context
    
    def _check_data_quality(self, context: DatasetContext):
        """Check for common data quality issues"""
        
        constant_cols = [col for col, info in context.schema_info.items() 
                        if info['is_constant']]
        if constant_cols:
            context.add_warning(f"Constant columns detected: {constant_cols}")
            
        high_missing_cols = [col for col, info in context.schema_info.items() 
                           if info['missing_pct'] > 50]
        if high_missing_cols:
            context.add_warning(f"High missing data (>50%): {high_missing_cols}")
            
        id_like_cols = [col for col, info in context.schema_info.items() 
                       if info['is_id_like']]
        if id_like_cols:
            context.add_warning(f"Potential ID columns (high cardinality): {id_like_cols}")
            
        # Check for duplicate rows
        dup_rows = context.df.duplicated().sum()
        if dup_rows > 0:
            context.add_warning(f"Duplicate rows found: {dup_rows}")
            
        # Check for duplicate columns
        dup_cols = []
        for i, col1 in enumerate(context.df.columns):
            for col2 in context.df.columns[i+1:]:
                if context.df[col1].equals(context.df[col2]):
                    dup_cols.append((col1, col2))
        if dup_cols:
            context.add_warning(f"Duplicate columns found: {dup_cols}")
    
    def _basic_cleanup(self, df: pd.DataFrame, context: DatasetContext) -> pd.DataFrame:
        """Perform basic cleanup operations"""
        
        # Remove constant columns (optional - could be configurable)
        constant_cols = [col for col, info in context.schema_info.items() 
                        if info['is_constant']]
        if constant_cols:
            df = df.drop(columns=constant_cols)
            context.add_warning(f"Dropped constant columns: {constant_cols}")
            
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            context.add_warning(f"Removed {initial_rows - len(df)} duplicate rows")
            
        return df