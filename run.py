"""
Startup script for Deepfake Detection Platform
"""

import sys
import os
import uvicorn
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
# Ensure relative paths in settings work consistently for reloader subprocesses.
Path(project_root).resolve()

from app.core.config import settings
from app.core.logging import logger


def create_directories():
    """Create necessary directories"""
    directories = [
        settings.UPLOAD_DIR,
        settings.DATA_DIR,
        settings.MODEL_DIR,
        settings.LOG_DIR,
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory created/verified: {directory}")


def main():
    """Main entry point"""
    # Keep cwd stable so relative paths (uploads/logs/models/db) resolve correctly.
    os.chdir(project_root)

    logger.info("Starting Deepfake Detection Platform", version=settings.APP_VERSION)
    
    # Create necessary directories
    create_directories()
    
    reload_enabled = bool(settings.RELOAD)
    workers = int(settings.WORKERS)

    # Watchfiles reload can be unstable on Windows in some environments
    # (continuous file-change events causing reloader loops).
    if reload_enabled and os.name == "nt" and os.getenv("FORCE_RELOAD", "0") != "1":
        logger.warning(
            "RELOAD disabled on Windows for stability; set FORCE_RELOAD=1 to force enable"
        )
        reload_enabled = False

    # Uvicorn does not support reload with multiple workers.
    if reload_enabled and workers > 1:
        logger.warning(
            "RELOAD is enabled; overriding WORKERS to 1",
            configured_workers=workers,
        )
        workers = 1

    uvicorn_kwargs = {
        "app": "app.main:app",
        "host": settings.HOST,
        "port": settings.PORT,
        "reload": reload_enabled,
        "workers": workers,
        "log_level": settings.LOG_LEVEL.lower(),
        "access_log": True,
    }

    if reload_enabled:
        # Prevent file writes in runtime directories from triggering reload loops.
        uvicorn_kwargs["reload_excludes"] = [
            "logs/*",
            "uploads/*",
            "data/*",
            "models/*",
            "*.log",
            "*.db",
        ]

    # Run the application; fall back if current uvicorn lacks reload_excludes.
    try:
        uvicorn.run(**uvicorn_kwargs)
    except TypeError as exc:
        if "reload_excludes" in str(exc):
            logger.warning(
                "Current uvicorn does not support reload_excludes; retrying with reload disabled"
            )
            uvicorn_kwargs.pop("reload_excludes", None)
            uvicorn_kwargs["reload"] = False
            uvicorn.run(**uvicorn_kwargs)
        else:
            raise


if __name__ == "__main__":
    main()
