"""
Startup script for Deepfake Detection Platform
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.core.config import settings
from app.core.logging import logger


def create_directories():
    """Create necessary directories"""
    directories = [
        settings.UPLOAD_DIR,
        settings.DATA_DIR,
        settings.MODELS_DIR,
        "logs",
        "alembic/versions"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory created/verified: {directory}")


def main():
    """Main entry point"""
    logger.info("Starting Deepfake Detection Platform", version=settings.APP_VERSION)
    
    # Create necessary directories
    create_directories()
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )


if __name__ == "__main__":
    main()
