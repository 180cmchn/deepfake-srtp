"""
API routes module for deepfake detection platform
"""

from fastapi import APIRouter
from .detection import router as detection_router
from .training import router as training_router
from .models import router as models_router
from .datasets import router as datasets_router

api_router = APIRouter()

# Include all route modules
api_router.include_router(detection_router, prefix="/detection", tags=["detection"])
api_router.include_router(training_router, prefix="/training", tags=["training"])
api_router.include_router(models_router, prefix="/models", tags=["models"])
api_router.include_router(datasets_router, prefix="/datasets", tags=["datasets"])
