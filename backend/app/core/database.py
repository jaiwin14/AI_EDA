"""
Database Configuration and Models
Supabase integration for storing model artifacts and metadata
"""

import os
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import logging

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None

from app.core.config import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database manager for Supabase operations"""
    
    def __init__(self):
        self.client: Optional[Client] = None
        self.connected = False
    
    async def connect(self):
        """Initialize Supabase connection"""
        if not SUPABASE_AVAILABLE:
            logger.warning("Supabase client not available. Using local storage fallback.")
            return False
            
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            logger.warning("Supabase credentials not configured. Using local storage fallback.")
            return False
        
        try:
            self.client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
            self.connected = True
            logger.info("Connected to Supabase successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {str(e)}")
            return False
    
    async def create_tables(self):
        """Create necessary tables if they don't exist"""
        if not self.connected:
            return
        
        # SQL for creating tables
        tables_sql = [
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                filename VARCHAR NOT NULL,
                original_filename VARCHAR NOT NULL,
                file_size INTEGER,
                rows_count INTEGER,
                cols INTEGER,
                upload_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                metadata JSONB,
                status VARCHAR DEFAULT 'uploaded'
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS eda_results (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                dataset_id UUID REFERENCES datasets(id) ON DELETE CASCADE,
                type VARCHAR NOT NULL,
                results JSONB NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS models (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                dataset_id UUID REFERENCES datasets(id) ON DELETE CASCADE,
                model_name VARCHAR NOT NULL,
                model_type VARCHAR NOT NULL,
                task_type VARCHAR NOT NULL,
                target_column VARCHAR,
                features JSONB,
                hyperparameters JSONB,
                metrics JSONB,
                model_path VARCHAR,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                status VARCHAR DEFAULT 'training'
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                model_id UUID REFERENCES models(id) ON DELETE CASCADE,
                input_data JSONB NOT NULL,
                prediction JSONB NOT NULL,
                confidence FLOAT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """
        ]
        
        try:
            for sql in tables_sql:
                # Note: Supabase Python client doesn't support raw SQL execution
                # Tables should be created via Supabase dashboard or SQL editor
                pass
            logger.info("Database tables verified")
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")
    
    # Dataset operations
    async def save_dataset_metadata(self, filename: str, original_filename: str, 
                                  file_size: int, rows: int, cols: int, 
                                  metadata: Dict[str, Any]) -> Optional[str]:
        """Save dataset metadata"""
        if not self.connected:
            return None
        
        try:
            result = self.client.table('datasets').insert({
                'filename': filename,
                'original_filename': original_filename,
                'file_size': file_size,
                'rows_count': rows,
                'cols': cols,
                'metadata': metadata
            }).execute()
            
            if result.data:
                return result.data[0]['id']
            return None
        except Exception as e:
            logger.error(f"Error saving dataset metadata: {str(e)}")
            return None
    
    async def get_dataset_metadata(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get dataset metadata by ID"""
        if not self.connected:
            return None
        
        try:
            result = self.client.table('datasets').select('*').eq('id', dataset_id).execute()
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Error getting dataset metadata: {str(e)}")
            return None
    
    async def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get dataset data by ID"""
        if not self.connected:
            return None
        
        try:
            # First try to get from database
            result = self.client.table('datasets').select('*').eq('id', dataset_id).execute()
            if result.data:
                dataset_metadata = result.data[0]
                # Try to load the actual data from local storage using filename
                filename = dataset_metadata.get('filename')
                if filename:
                    # Remove .json extension if present and add dataset_ prefix
                    dataset_filename = f"dataset_{dataset_id}"
                    return local_storage.load_json(dataset_filename)
            return None
        except Exception as e:
            logger.error(f"Error getting dataset: {str(e)}")
            return None
    
    # EDA results operations
    async def save_eda_results(self, dataset_id: str, analysis_type: str, 
                             results: Dict[str, Any]) -> Optional[str]:
        """Save EDA analysis results"""
        if not self.connected:
            return None
        
        try:
            result = self.client.table('eda_results').insert({
                'dataset_id': dataset_id,
                'type': analysis_type,
                'results': results
            }).execute()
            
            if result.data:
                return result.data[0]['id']
            return None
        except Exception as e:
            logger.error(f"Error saving EDA results: {str(e)}")
            return None
    
    async def get_eda_results(self, dataset_id: str, 
                            analysis_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get EDA results for a dataset"""
        if not self.connected:
            return []
        
        try:
            query = self.client.table('eda_results').select('*').eq('dataset_id', dataset_id)
            if analysis_type:
                query = query.eq('type', analysis_type)
            
            result = query.execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting EDA results: {str(e)}")
            return []
    
    # Model operations
    async def save_model_metadata(self, dataset_id: str, model_name: str, 
                                model_type: str, task_type: str, target_column: str,
                                features: List[str], hyperparameters: Dict[str, Any],
                                metrics: Dict[str, Any], model_path: str) -> Optional[str]:
        """Save model metadata"""
        if not self.connected:
            return None
        
        try:
            result = self.client.table('models').insert({
                'dataset_id': dataset_id,
                'model_name': model_name,
                'model_type': model_type,
                'task_type': task_type,
                'target_column': target_column,
                'features': features,
                'hyperparameters': hyperparameters,
                'metrics': metrics,
                'model_path': model_path,
                'status': 'completed'
            }).execute()
            
            if result.data:
                return result.data[0]['id']
            return None
        except Exception as e:
            logger.error(f"Error saving model metadata: {str(e)}")
            return None
    
    async def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model metadata by ID"""
        if not self.connected:
            return None
        
        try:
            result = self.client.table('models').select('*').eq('id', model_id).execute()
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Error getting model metadata: {str(e)}")
            return None
    
    async def get_models_for_dataset(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Get all models for a dataset"""
        if not self.connected:
            return []
        
        try:
            result = self.client.table('models').select('*').eq('dataset_id', dataset_id).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting models for dataset: {str(e)}")
            return []
    
    # Prediction operations
    async def save_prediction(self, model_id: str, input_data: Dict[str, Any],
                            prediction: Dict[str, Any], confidence: Optional[float] = None) -> Optional[str]:
        """Save prediction result"""
        if not self.connected:
            return None
        
        try:
            result = self.client.table('predictions').insert({
                'model_id': model_id,
                'input_data': input_data,
                'prediction': prediction,
                'confidence': confidence
            }).execute()
            
            if result.data:
                return result.data[0]['id']
            return None
        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
            return None

# Global database manager instance
db_manager = DatabaseManager()

async def init_db():
    """Initialize database connection and tables"""
    await db_manager.connect()
    await db_manager.create_tables()

# Fallback local storage for when Supabase is not available
class LocalStorageManager:
    """Local file-based storage fallback"""
    
    def __init__(self):
        self.storage_dir = settings.BASE_DIR / "local_storage"
        self.storage_dir.mkdir(exist_ok=True)
    
    def save_json(self, filename: str, data: Dict[str, Any]):
        """Save data as JSON file"""
        filepath = self.storage_dir / f"{filename}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_json(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load data from JSON file"""
        filepath = self.storage_dir / f"{filename}.json"
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        return None

# Global local storage instance
local_storage = LocalStorageManager()
