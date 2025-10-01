"""
Training API routes for deepfake detection platform
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional

from app.core.database import get_db
from app.core.logging import logger
from app.schemas.training import (
    TrainingJobCreate, TrainingJobResponse, TrainingJobList,
    TrainingJobUpdate, TrainingProgress, TrainingMetrics
)
from app.services.training_service import TrainingService

router = APIRouter()


@router.post("/jobs", response_model=TrainingJobResponse)
async def create_training_job(
    job: TrainingJobCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create a new training job"""
    try:
        training_service = TrainingService(db)
        result = await training_service.create_job(job, background_tasks)
        return result
    except Exception as e:
        logger.error("Failed to create training job", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create training job")


@router.get("/jobs", response_model=TrainingJobList)
async def get_training_jobs(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    model_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get training jobs list"""
    try:
        training_service = TrainingService(db)
        result = await training_service.get_jobs(skip, limit, status, model_type)
        return result
    except Exception as e:
        logger.error("Failed to get training jobs", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get training jobs")


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(job_id: int, db: Session = Depends(get_db)):
    """Get training job by ID"""
    try:
        training_service = TrainingService(db)
        result = await training_service.get_job(job_id)
        if not result:
            raise HTTPException(status_code=404, detail="Training job not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get training job", error=str(e), job_id=job_id)
        raise HTTPException(status_code=500, detail="Failed to get training job")


@router.put("/jobs/{job_id}", response_model=TrainingJobResponse)
async def update_training_job(
    job_id: int,
    job_update: TrainingJobUpdate,
    db: Session = Depends(get_db)
):
    """Update training job"""
    try:
        training_service = TrainingService(db)
        result = await training_service.update_job(job_id, job_update)
        if not result:
            raise HTTPException(status_code=404, detail="Training job not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update training job", error=str(e), job_id=job_id)
        raise HTTPException(status_code=500, detail="Failed to update training job")


@router.delete("/jobs/{job_id}")
async def delete_training_job(job_id: int, db: Session = Depends(get_db)):
    """Delete training job"""
    try:
        training_service = TrainingService(db)
        success = await training_service.delete_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail="Training job not found")
        return {"message": "Training job deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete training job", error=str(e), job_id=job_id)
        raise HTTPException(status_code=500, detail="Failed to delete training job")


@router.get("/jobs/{job_id}/progress", response_model=TrainingProgress)
async def get_training_progress(job_id: int, db: Session = Depends(get_db)):
    """Get training job progress"""
    try:
        training_service = TrainingService(db)
        progress = await training_service.get_progress(job_id)
        if not progress:
            raise HTTPException(status_code=404, detail="Training job not found")
        return progress
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get training progress", error=str(e), job_id=job_id)
        raise HTTPException(status_code=500, detail="Failed to get training progress")


@router.get("/metrics", response_model=TrainingMetrics)
async def get_training_metrics(db: Session = Depends(get_db)):
    """Get training metrics"""
    try:
        training_service = TrainingService(db)
        metrics = await training_service.get_metrics()
        return metrics
    except Exception as e:
        logger.error("Failed to get training metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get training metrics")
