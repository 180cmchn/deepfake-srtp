"""
FastAPI application for deepfake detection platform
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import structlog
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from app.api.routes import api_router
from app.core.config import settings
from app.core.database import create_tables, test_connection
from app.core.logging import logger

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="A modular deepfake detection system",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    redirect_slashes=False
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix=settings.API_V1_STR)

# Mount static directories
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")
app.mount("/models", StaticFiles(directory=settings.MODEL_DIR), name="models")

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("Deepfake Detection Platform starting up", 
                project_name=settings.PROJECT_NAME,
                database_url=settings.DATABASE_URL.split('/')[-1])
    
    # Create necessary directories
    import os
    for directory in [settings.UPLOAD_DIR, settings.MODEL_DIR, settings.LOG_DIR, settings.DATA_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Test database connection
    if test_connection():
        logger.info("Database connection verified")
        # Create database tables
        create_tables()
        logger.info("Database initialization completed")
    else:
        logger.error("Database connection failed - application may not work properly")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("Deepfake Detection Platform shutting down")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": f"{settings.PROJECT_NAME} API", 
        "version": "1.0.0",
        "docs_url": f"{settings.API_V1_STR}/docs",
        "supported_models": settings.SUPPORTED_MODELS
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    db_status = "healthy" if test_connection() else "unhealthy"
    return {
        "status": "healthy",
        "database": db_status,
        "project": settings.PROJECT_NAME,
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
