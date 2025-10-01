"""
Model management API routes for deepfake detection platform
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional

from app.core.database import get_db
from app.core.logging import logger
from app.schemas.models import (
    ModelCreate, ModelResponse, ModelList, ModelUpdate,
    ModelStatistics, ModelDeployment
)
from app.services.model_service import ModelService

router = APIRouter()


@router.post("/", response_model=ModelResponse)
async def create_model(model: ModelCreate, db: Session = Depends(get_db)):
    """Create a new model registry entry"""
    try:
        model_service = ModelService(db)
        result = await model_service.create_model(model)
        return result
    except Exception as e:
        logger.error("Failed to create model", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create model")


@router.get("/", response_model=ModelList)
async def get_models(
    skip: int = 0,
    limit: int = 100,
    model_type: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get models list"""
    try:
        model_service = ModelService(db)
        result = await model_service.get_models(skip, limit, model_type, status)
        return result
    except Exception as e:
        logger.error("Failed to get models", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get models")


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(model_id: int, db: Session = Depends(get_db)):
    """Get model by ID"""
    try:
        model_service = ModelService(db)
        result = await model_service.get_model(model_id)
        if not result:
            raise HTTPException(status_code=404, detail="Model not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get model", error=str(e), model_id=model_id)
        raise HTTPException(status_code=500, detail="Failed to get model")


@router.put("/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: int,
    model_update: ModelUpdate,
    db: Session = Depends(get_db)
):
    """Update model"""
    try:
        model_service = ModelService(db)
        result = await model_service.update_model(model_id, model_update)
        if not result:
            raise HTTPException(status_code=404, detail="Model not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update model", error=str(e), model_id=model_id)
        raise HTTPException(status_code=500, detail="Failed to update model")


@router.delete("/{model_id}")
async def delete_model(model_id: int, db: Session = Depends(get_db)):
    """Delete model"""
    try:
        model_service = ModelService(db)
        success = await model_service.delete_model(model_id)
        if not success:
            raise HTTPException(status_code=404, detail="Model not found")
        return {"message": "Model deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete model", error=str(e), model_id=model_id)
        raise HTTPException(status_code=500, detail="Failed to delete model")


@router.post("/{model_id}/deploy", response_model=ModelDeployment)
async def deploy_model(
    model_id: int,
    deployment_config: dict,
    db: Session = Depends(get_db)
):
    """Deploy model"""
    try:
        model_service = ModelService(db)
        result = await model_service.deploy_model(model_id, deployment_config)
        return result
    except Exception as e:
        logger.error("Failed to deploy model", error=str(e), model_id=model_id)
        raise HTTPException(status_code=500, detail="Failed to deploy model")


@router.get("/statistics/overview", response_model=ModelStatistics)
async def get_model_statistics(db: Session = Depends(get_db)):
    """Get model statistics"""
    try:
        model_service = ModelService(db)
        stats = await model_service.get_statistics()
        return stats
    except Exception as e:
        logger.error("Failed to get model statistics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get model statistics")
