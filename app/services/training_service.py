"""
Training service for deepfake detection platform
"""

import os
import time
import asyncio
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from fastapi import BackgroundTasks

from app.core.database import get_db_session
from app.core.logging import logger
from app.core.config import settings
from app.models.database_models import TrainingJob, ModelRegistry
from app.schemas.training import (
    TrainingJobCreate, TrainingJobResponse, TrainingJobList,
    TrainingJobUpdate, TrainingProgress, TrainingMetrics, JobStatus
)


class TrainingService:
    """Service for handling model training operations"""
    
    def __init__(self, db: Session):
        self.db = db
        self._active_jobs = {}
    
    async def create_job(self, job: TrainingJobCreate, background_tasks: BackgroundTasks) -> TrainingJobResponse:
        """Create a new training job"""
        try:
            # Check concurrent job limit
            active_count = self.db.query(TrainingJob).filter(
                TrainingJob.status.in_([JobStatus.PENDING.value, JobStatus.RUNNING.value]),
                TrainingJob.del_flag == 0
            ).count()
            
            if active_count >= settings.MAX_CONCURRENT_TRAINING_JOBS:
                raise ValueError(f"Maximum concurrent training jobs ({settings.MAX_CONCURRENT_TRAINING_JOBS}) reached")
            
            # Create training job
            db_job = TrainingJob(
                name=job.name,
                description=job.description,
                model_type=job.model_type,
                dataset_path=job.dataset_path,
                epochs=job.parameters.epochs,
                learning_rate=job.parameters.learning_rate,
                batch_size=job.parameters.batch_size
            )
            
            self.db.add(db_job)
            self.db.commit()
            self.db.refresh(db_job)
            
            # Start training in background
            background_tasks.add_task(self._run_training, db_job.id, job.parameters.dict())
            
            logger.info("Training job created", job_id=db_job.id, model_type=job.model_type)
            
            return self._db_to_response(db_job)
            
        except Exception as e:
            logger.error("Failed to create training job", error=str(e))
            self.db.rollback()
            raise
    
    async def get_jobs(
        self,
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> TrainingJobList:
        """Get training jobs list"""
        try:
            query = self.db.query(TrainingJob).filter(TrainingJob.del_flag == 0)
            
            if status:
                query = query.filter(TrainingJob.status == status)
            
            if model_type:
                query = query.filter(TrainingJob.model_type == model_type)
            
            total = query.count()
            results = query.order_by(TrainingJob.created_at.desc()).offset(skip).limit(limit).all()
            
            jobs = [self._db_to_response(job) for job in results]
            
            return TrainingJobList(
                jobs=jobs,
                total=total,
                page=skip // limit + 1,
                size=limit,
                pages=(total + limit - 1) // limit
            )
            
        except Exception as e:
            logger.error("Failed to get training jobs", error=str(e))
            raise
    
    async def get_job(self, job_id: int) -> Optional[TrainingJobResponse]:
        """Get training job by ID"""
        try:
            job = self.db.query(TrainingJob).filter(
                TrainingJob.id == job_id,
                TrainingJob.del_flag == 0
            ).first()
            
            if not job:
                return None
            
            return self._db_to_response(job)
            
        except Exception as e:
            logger.error("Failed to get training job", error=str(e), job_id=job_id)
            raise
    
    async def update_job(self, job_id: int, job_update: TrainingJobUpdate) -> Optional[TrainingJobResponse]:
        """Update training job"""
        try:
            job = self.db.query(TrainingJob).filter(
                TrainingJob.id == job_id,
                TrainingJob.del_flag == 0
            ).first()
            
            if not job:
                return None
            
            # Update fields
            update_data = job_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                if hasattr(job, field):
                    setattr(job, field, value)
            
            self.db.commit()
            self.db.refresh(job)
            
            logger.info("Training job updated", job_id=job_id)
            
            return self._db_to_response(job)
            
        except Exception as e:
            logger.error("Failed to update training job", error=str(e), job_id=job_id)
            self.db.rollback()
            raise
    
    async def delete_job(self, job_id: int) -> bool:
        """Delete training job"""
        try:
            job = self.db.query(TrainingJob).filter(
                TrainingJob.id == job_id,
                TrainingJob.del_flag == 0
            ).first()
            
            if not job:
                return False
            
            # Cancel if running
            if job.status == JobStatus.RUNNING.value:
                await self._cancel_training(job_id)
            
            job.del_flag = 1
            self.db.commit()
            
            logger.info("Training job deleted", job_id=job_id)
            return True
            
        except Exception as e:
            logger.error("Failed to delete training job", error=str(e), job_id=job_id)
            self.db.rollback()
            raise
    
    async def get_progress(self, job_id: int) -> Optional[TrainingProgress]:
        """Get training job progress"""
        try:
            job = self.db.query(TrainingJob).filter(
                TrainingJob.id == job_id,
                TrainingJob.del_flag == 0
            ).first()
            
            if not job:
                return None
            
            return TrainingProgress(
                job_id=job.id,
                status=JobStatus(job.status),
                progress=job.progress,
                current_epoch=None,  # TODO: Track current epoch
                total_epochs=job.epochs,
                current_loss=None,  # TODO: Track current loss
                current_accuracy=None,  # TODO: Track current accuracy
                estimated_time_remaining=None,  # TODO: Calculate ETA
                message=None
            )
            
        except Exception as e:
            logger.error("Failed to get training progress", error=str(e), job_id=job_id)
            raise
    
    async def get_metrics(self) -> TrainingMetrics:
        """Get training metrics"""
        try:
            # Get job counts
            total_jobs = self.db.query(TrainingJob).filter(TrainingJob.del_flag == 0).count()
            active_jobs = self.db.query(TrainingJob).filter(
                TrainingJob.status.in_([JobStatus.PENDING.value, JobStatus.RUNNING.value]),
                TrainingJob.del_flag == 0
            ).count()
            completed_jobs = self.db.query(TrainingJob).filter(
                TrainingJob.status == JobStatus.COMPLETED.value,
                TrainingJob.del_flag == 0
            ).count()
            failed_jobs = self.db.query(TrainingJob).filter(
                TrainingJob.status == JobStatus.FAILED.value,
                TrainingJob.del_flag == 0
            ).count()
            
            # Calculate average accuracy
            completed_with_accuracy = self.db.query(TrainingJob.accuracy).filter(
                TrainingJob.status == JobStatus.COMPLETED.value,
                TrainingJob.accuracy.isnot(None),
                TrainingJob.del_flag == 0
            ).all()
            average_accuracy = sum(acc[0] for acc in completed_with_accuracy) / len(completed_with_accuracy) if completed_with_accuracy else None
            
            # Calculate average training time
            completed_with_time = self.db.query(
                TrainingJob.started_at, TrainingJob.completed_at
            ).filter(
                TrainingJob.status == JobStatus.COMPLETED.value,
                TrainingJob.started_at.isnot(None),
                TrainingJob.completed_at.isnot(None),
                TrainingJob.del_flag == 0
            ).all()
            
            training_times = []
            for started, completed in completed_with_time:
                if started and completed:
                    training_times.append((completed - started).total_seconds())
            
            average_training_time = sum(training_times) / len(training_times) if training_times else None
            
            # Jobs by model type
            jobs_by_model_type = {}
            for model_type in settings.SUPPORTED_MODELS:
                count = self.db.query(TrainingJob).filter(
                    TrainingJob.model_type == model_type,
                    TrainingJob.del_flag == 0
                ).count()
                if count > 0:
                    jobs_by_model_type[model_type] = count
            
            # Jobs by status
            jobs_by_status = {
                "pending": self.db.query(TrainingJob).filter(
                    TrainingJob.status == JobStatus.PENDING.value,
                    TrainingJob.del_flag == 0
                ).count(),
                "running": self.db.query(TrainingJob).filter(
                    TrainingJob.status == JobStatus.RUNNING.value,
                    TrainingJob.del_flag == 0
                ).count(),
                "completed": completed_jobs,
                "failed": failed_jobs,
                "cancelled": self.db.query(TrainingJob).filter(
                    TrainingJob.status == JobStatus.CANCELLED.value,
                    TrainingJob.del_flag == 0
                ).count()
            }
            
            success_rate = completed_jobs / total_jobs if total_jobs > 0 else 0.0
            
            return TrainingMetrics(
                total_jobs=total_jobs,
                active_jobs=active_jobs,
                completed_jobs=completed_jobs,
                failed_jobs=failed_jobs,
                average_accuracy=average_accuracy,
                average_training_time=average_training_time,
                success_rate=success_rate,
                jobs_by_model_type=jobs_by_model_type,
                jobs_by_status=jobs_by_status
            )
            
        except Exception as e:
            logger.error("Failed to get training metrics", error=str(e))
            raise
    
    # Private methods
    
    async def _run_training(self, job_id: int, parameters: Dict[str, Any]):
        """Run training job in background"""
        try:
            # Update job status to running
            job = self.db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if not job:
                return
            
            job.status = JobStatus.RUNNING.value
            job.started_at = time.time()
            self.db.commit()
            
            # TODO: Implement actual training logic
            # This would involve:
            # 1. Loading dataset
            # 2. Creating model
            # 3. Training loop
            # 4. Saving model
            # 5. Updating progress
            
            # Simulate training for now
            await self._simulate_training(job_id, parameters)
            
        except Exception as e:
            logger.error("Training failed", error=str(e), job_id=job_id)
            
            # Update job status to failed
            job = self.db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if job:
                job.status = JobStatus.FAILED.value
                job.error_message = str(e)
                job.completed_at = time.time()
                self.db.commit()
    
    async def _simulate_training(self, job_id: int, parameters: Dict[str, Any]):
        """Simulate training process"""
        epochs = parameters.get("epochs", 50)
        
        for epoch in range(epochs):
            # Simulate training time
            await asyncio.sleep(0.1)
            
            # Update progress
            progress = (epoch + 1) / epochs * 100
            
            job = self.db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if job:
                job.progress = progress
                
                # Simulate improving accuracy
                if epoch > 0:
                    job.accuracy = min(0.95, 0.5 + (epoch / epochs) * 0.45)
                    job.loss = max(0.1, 1.0 - (epoch / epochs) * 0.9)
                
                self.db.commit()
        
        # Mark as completed
        job = self.db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if job:
            job.status = JobStatus.COMPLETED.value
            job.progress = 100.0
            job.completed_at = time.time()
            job.accuracy = 0.95
            job.loss = 0.1
            job.model_path = f"models/trained_model_{job_id}.pth"
            self.db.commit()
    
    async def _cancel_training(self, job_id: int):
        """Cancel running training job"""
        # TODO: Implement actual training cancellation
        # This would involve stopping the training process
        if job_id in self._active_jobs:
            del self._active_jobs[job_id]
    
    def _db_to_response(self, job: TrainingJob) -> TrainingJobResponse:
        """Convert database model to response schema"""
        from app.schemas.training import TrainingParameters, TrainingResults
        
        parameters = TrainingParameters(
            epochs=job.epochs,
            learning_rate=job.learning_rate,
            batch_size=job.batch_size
        )
        
        results = None
        if job.accuracy is not None:
            results = TrainingResults(
                accuracy=job.accuracy,
                loss=job.loss,
                model_path=job.model_path
            )
        
        return TrainingJobResponse(
            id=job.id,
            name=job.name,
            description=job.description,
            model_type=job.model_type,
            dataset_path=job.dataset_path,
            parameters=parameters,
            status=JobStatus(job.status),
            progress=job.progress,
            results=results,
            created_at=job.created_at,
            updated_at=job.updated_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message
        )
