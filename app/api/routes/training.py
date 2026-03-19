"""
Training API routes for deepfake detection platform
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Header, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from app.core.database import get_db
from app.core.logging import logger
from app.core.auth import get_current_user, require_admin
from app.models.database_models import ModelRegistry
from app.schemas.training import (
    TrainingJobCreate,
    TrainingJobResponse,
    TrainingJobList,
    TrainingJobUpdate,
    TrainingProgress,
    TrainingMetrics,
)

router = APIRouter()


@router.post("/jobs", response_model=TrainingJobResponse)
async def create_training_job(
    job: TrainingJobCreate,
    background_tasks: BackgroundTasks,
    auto_start: bool = Query(
        False, description="Automatically start the job after creation"
    ),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    """Create a new training job"""
    try:
        from app.services.training_service import TrainingService

        training_service = TrainingService(db)
        result = await training_service.create_job(
            job, background_tasks, auto_start=auto_start
        )

        logger.info(
            "Training job created",
            job_id=result.id,
            job_name=result.name,
            created_by=current_user,
            auto_start=auto_start,
        )

        return result
    except Exception as e:
        logger.error("Failed to create training job", error=str(e), user=current_user)
        raise HTTPException(
            status_code=500, detail=f"Failed to create training job: {str(e)}"
        )


@router.get("/jobs", response_model=TrainingJobList)
async def get_training_jobs(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(
        100, ge=1, le=1000, description="Maximum number of records to return"
    ),
    status: Optional[str] = Query(None, description="Filter by job status"),
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    created_by: Optional[str] = Query(None, description="Filter by job creator"),
    search: Optional[str] = Query(
        None, description="Search in job name and description"
    ),
    order_by: str = Query("created_at", description="Field to order by"),
    order_desc: bool = Query(True, description="Order in descending order"),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    """Get training jobs list with filtering, searching, and pagination"""
    try:
        from app.services.training_service import TrainingService

        training_service = TrainingService(db)
        result = await training_service.get_jobs(
            skip=skip,
            limit=limit,
            status=status,
            model_type=model_type,
            created_by=created_by,
            search=search,
            order_by=order_by,
            order_desc=order_desc,
        )
        return result
    except Exception as e:
        logger.error("Failed to get training jobs", error=str(e), user=current_user)
        raise HTTPException(status_code=500, detail="Failed to get training jobs")


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(job_id: int, db: Session = Depends(get_db)):
    """Get training job by ID"""
    try:
        from app.services.training_service import TrainingService

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
    job_id: int, job_update: TrainingJobUpdate, db: Session = Depends(get_db)
):
    """Update training job"""
    try:
        from app.services.training_service import TrainingService

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
        from app.services.training_service import TrainingService

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
        from app.services.training_service import TrainingService

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
        from app.services.training_service import TrainingService

        training_service = TrainingService(db)
        metrics = await training_service.get_metrics()
        return metrics
    except Exception as e:
        logger.error("Failed to get training metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get training metrics")


@router.post("/jobs/{job_id}/start")
async def start_training_job(
    job_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    """Start a training job manually"""
    try:
        from app.services.training_service import TrainingService

        training_service = TrainingService(db)

        # Get the job first to check its current status
        job = await training_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")

        # Start the job
        success = await training_service.start_job(
            job_id, background_tasks, current_user
        )

        if not success:
            raise HTTPException(
                status_code=400, detail="Cannot start job in current status"
            )

        logger.info(
            "Training job started",
            job_id=job_id,
            job_name=job.name,
            started_by=current_user,
        )

        return {
            "message": f"Training job '{job.name}' (ID: {job_id}) started successfully",
            "job_id": job_id,
            "job_name": job.name,
            "started_by": current_user,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to start training job",
            error=str(e),
            job_id=job_id,
            user=current_user,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to start training job: {str(e)}"
        )


@router.post("/jobs/{job_id}/stop")
async def stop_training_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    """Stop a running training job"""
    try:
        from app.services.training_service import TrainingService

        training_service = TrainingService(db)

        # Get the job first to check its current status
        job = await training_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")

        # Stop the job
        success = await training_service.stop_job(job_id, current_user)

        if not success:
            raise HTTPException(
                status_code=400, detail="Cannot stop job in current status"
            )

        logger.info(
            "Training job stopped",
            job_id=job_id,
            job_name=job.name,
            stopped_by=current_user,
        )

        return {
            "message": f"Training job '{job.name}' (ID: {job_id}) stopped successfully",
            "job_id": job_id,
            "job_name": job.name,
            "stopped_by": current_user,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to stop training job",
            error=str(e),
            job_id=job_id,
            user=current_user,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to stop training job: {str(e)}"
        )


@router.get("/jobs/{job_id}/logs")
async def get_training_job_logs(
    job_id: int,
    tail_lines: int = Query(
        100, ge=0, le=10000, description="Number of lines to return from end of log"
    ),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    """Get logs for a training job"""
    try:
        from app.services.training_service import TrainingService

        training_service = TrainingService(db)
        job = await training_service.get_job(job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")

        logs = await training_service.get_job_logs(job_id, tail_lines)

        return {
            "job_id": job_id,
            "job_name": job.name,
            "logs": logs,
            "tail_lines": tail_lines,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get training job logs",
            error=str(e),
            job_id=job_id,
            user=current_user,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to get training job logs: {str(e)}"
        )


@router.post("/jobs/{job_id}/model/retain")
async def retain_training_job_model(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    """Confirm retaining the trained model file after reviewing training results."""
    try:
        from app.services.training_service import TrainingService

        training_service = TrainingService(db)
        model_path = await training_service.retain_model_file(job_id, current_user)

        if model_path is None:
            raise HTTPException(status_code=404, detail="Training job not found")

        model_record = (
            db.query(ModelRegistry)
            .filter(
                ModelRegistry.training_job_id == job_id, ModelRegistry.del_flag == 0
            )
            .first()
        )

        return {
            "message": "Model file retention confirmed",
            "job_id": job_id,
            "model_path": model_path,
            "model_id": model_record.id if model_record else None,
            "model_name": model_record.name if model_record else None,
            "decided_by": current_user,
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            "Failed to retain training job model",
            error=str(e),
            job_id=job_id,
            user=current_user,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to retain training job model: {str(e)}"
        )


@router.delete("/jobs/{job_id}/model")
async def discard_training_job_model(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    """Discard the trained model file after reviewing training results."""
    try:
        from app.services.training_service import TrainingService

        training_service = TrainingService(db)
        success = await training_service.discard_model_file(job_id, current_user)

        if not success:
            raise HTTPException(status_code=404, detail="Training job not found")

        return {
            "message": "Model file discarded",
            "job_id": job_id,
            "discarded_by": current_user,
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            "Failed to discard training job model",
            error=str(e),
            job_id=job_id,
            user=current_user,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to discard training job model: {str(e)}"
        )


@router.get("/statistics")
async def get_training_job_statistics(
    db: Session = Depends(get_db), current_user: str = Depends(get_current_user)
):
    """Get training job statistics"""
    try:
        from app.services.training_service import TrainingService

        training_service = TrainingService(db)
        stats = await training_service.get_job_statistics()
        return stats

    except Exception as e:
        logger.error(
            "Failed to get training job statistics", error=str(e), user=current_user
        )
        raise HTTPException(
            status_code=500, detail="Failed to get training job statistics"
        )
