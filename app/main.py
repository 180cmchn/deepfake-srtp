"""
FastAPI application for deepfake detection platform
"""

from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import structlog
import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from app.api.routes import api_router
from app.core.config import settings
from app.core.database import (
    create_tables_with_retry,
    ensure_runtime_schema,
    get_database_health_snapshot,
    test_connection,
)
from app.core.logging import logger

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="A modular deepfake detection system",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    redirect_slashes=False,
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
    logger.info(
        "Deepfake Detection Platform starting up",
        project_name=settings.PROJECT_NAME,
        database_url=settings.DATABASE_URL.split("/")[-1],
    )

    # Create necessary directories
    import os

    for directory in [
        settings.UPLOAD_DIR,
        settings.MODEL_DIR,
        settings.LOG_DIR,
        settings.DATA_DIR,
    ]:
        os.makedirs(directory, exist_ok=True)

    if create_tables_with_retry():
        ensure_runtime_schema()

    skip_db_check = os.getenv("SKIP_STARTUP_DB_CHECK", "0") == "1"
    if skip_db_check:
        logger.warning("Skipping startup database check due to SKIP_STARTUP_DB_CHECK=1")
        return

    # Test database connection
    if test_connection():
        logger.info("Database connection verified")
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
        "docs_url": "/docs",
        "supported_models": settings.SUPPORTED_MODELS,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    db_health = get_database_health_snapshot()
    payload = {
        "status": db_health["status"],
        "timestamp": datetime.utcnow().isoformat(),
        "database": db_health["status"],
        "checks": {"database": db_health},
        "project": settings.PROJECT_NAME,
        "version": settings.APP_VERSION,
    }
    return JSONResponse(
        status_code=200 if db_health["healthy"] else 503,
        content=payload,
    )


if __name__ == "__main__":
    from run import main as run_main

    run_main()
