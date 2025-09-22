"""
FastAPI Main Application
Production-ready ML web app with automated EDA, model training, and predictions
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import uuid
import os
from typing import Dict, Any, List, Optional
import json

from app.api.upload import router as upload_router
from app.api.eda import router as eda_router
from app.api.models import router as models_router
from app.api.predictions import router as predictions_router
from app.api.explanations import router as explanations_router
from app.api.ai_insights import router as ai_insights_router
from app.core.config import settings
from app.core.database import init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI EDA & ML Platform",
    description="Production-ready ML web app with automated EDA, model training, and predictions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database and create necessary directories"""
    try:
        await init_db()
        
        # Create necessary directories
        directories = [
            settings.UPLOAD_DIR,
            settings.MODELS_DIR,
            settings.REPORTS_DIR,
            settings.TEMP_DIR
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI EDA & ML Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "upload": "/api/v1/upload",
            "eda": "/api/v1/eda",
            "models": "/api/v1/models",
            "predictions": "/api/v1/predict",
            "explanations": "/api/v1/explain",
            "ai_insights": "/api/v1/ai-insights"
        }
    }

# Include API routers
app.include_router(upload_router, prefix="/api/v1", tags=["upload"])
app.include_router(eda_router, prefix="/api/v1", tags=["eda"])
app.include_router(models_router, prefix="/api/v1", tags=["models"])
app.include_router(predictions_router, prefix="/api/v1", tags=["predictions"])
app.include_router(explanations_router, prefix="/api/v1", tags=["explanations"])
app.include_router(ai_insights_router, prefix="/api/v1", tags=["ai-insights"])

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for better error responses"""
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True if os.getenv("ENVIRONMENT") == "development" else False
    )
