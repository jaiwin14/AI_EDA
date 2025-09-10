"""
Application Configuration
Environment variables and settings management
"""

import os
from pathlib import Path
from typing import List
from pydantic import field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # App settings
    APP_NAME: str = "AI EDA & ML Platform"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    
    # API settings
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "https://your-netlify-app.netlify.app",  # Production frontend
    ]
    
    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v
    
    # Database settings (Supabase)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    
    # File storage settings
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    MODELS_DIR: Path = BASE_DIR / "models"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    TEMP_DIR: Path = BASE_DIR / "temp"
    
    # File upload limits
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_FILE_TYPES: List[str] = [".csv", ".xlsx", ".xls", ".parquet"]
    
    # ML settings
    MAX_ROWS_FOR_PROCESSING: int = 100000
    MODEL_TRAINING_TIMEOUT: int = 300  # 5 minutes
    RANDOM_STATE: int = 42
    
    # Gemini API settings
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemini-pro"
    MAX_TOKENS: int = 2048
    
    # Monitoring settings (Sentry)
    SENTRY_DSN: str = os.getenv("SENTRY_DSN", "")
    
    # Redis settings (for caching if needed)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore"
    }

# Global settings instance
settings = Settings()

# Validate critical settings
def validate_settings():
    """Validate critical settings on startup"""
    errors = []
    
    if not settings.GEMINI_API_KEY and settings.ENVIRONMENT == "production":
        errors.append("GEMINI_API_KEY is required in production")
    
    if not settings.DATABASE_URL and settings.ENVIRONMENT == "production":
        errors.append("DATABASE_URL is required in production")
    
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")

# Call validation
validate_settings()
