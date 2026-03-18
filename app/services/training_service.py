"""
Training service for deepfake detection platform
"""

import asyncio
from datetime import datetime
from pathlib import Path
import re
from typing import List, Optional, Dict, Any, Tuple

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sqlalchemy.orm import Session
from sqlalchemy import or_
from fastapi import BackgroundTasks

from app.core.database import get_db_session
from app.core.logging import logger
from app.core.config import settings
from app.models.database_models import TrainingJob
from app.schemas.training import (
    TrainingJobCreate,
    TrainingJobResponse,
    TrainingJobList,
    TrainingJobUpdate,
    TrainingProgress,
    TrainingMetrics,
    JobStatus,
)
from app.models.ml_models import create_model


FAKE_LABEL_KEYWORDS = {
    "fake",
    "deepfake",
    "manipulated",
    "forged",
    "tampered",
    "class1",
    "1",
}
REAL_LABEL_KEYWORDS = {
    "real",
    "authentic",
    "original",
    "genuine",
    "pristine",
    "class0",
    "0",
}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class TrainingCancelledError(Exception):
    """Raised when a training job is cancelled during execution."""


class ImageClassificationDataset(Dataset):
    """Simple image classification dataset for real/fake labels."""

    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label = self.samples[idx]
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class SequenceClassificationDataset(Dataset):
    """Dataset for video frame clips used by sequence models."""

    def __init__(self, samples: List[Tuple[List[str], int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        frame_paths, label = self.samples[idx]
        frames = []

        for frame_path in frame_paths:
            with Image.open(frame_path) as img:
                image = img.convert("RGB")
            if self.transform:
                image = self.transform(image)
            frames.append(image)

        clip = torch.stack(frames, dim=0)
        return clip, label


class TrainingService:
    """Service for handling model training operations"""

    _shared_active_jobs: Dict[int, Dict[str, Any]] = {}

    def __init__(self, db: Session):
        self.db = db
        self._active_jobs = self._shared_active_jobs

    async def create_job(
        self,
        job: TrainingJobCreate,
        background_tasks: BackgroundTasks,
        auto_start: bool = True,
    ) -> TrainingJobResponse:
        """Create a new training job"""
        try:
            # Check concurrent job limit
            active_count = (
                self.db.query(TrainingJob)
                .filter(
                    TrainingJob.status.in_(
                        [JobStatus.PENDING.value, JobStatus.RUNNING.value]
                    ),
                    TrainingJob.del_flag == 0,
                )
                .count()
            )

            if active_count >= settings.MAX_CONCURRENT_TRAINING_JOBS:
                raise ValueError(
                    f"Maximum concurrent training jobs ({settings.MAX_CONCURRENT_TRAINING_JOBS}) reached"
                )

            # Create training job
            db_job = TrainingJob(
                name=job.name,
                description=job.description,
                model_type=job.model_type,
                dataset_path=job.dataset_path,
                epochs=job.parameters.epochs,
                learning_rate=job.parameters.learning_rate,
                batch_size=job.parameters.batch_size,
            )

            self.db.add(db_job)
            self.db.commit()
            self.db.refresh(db_job)

            # Start training in background only when requested
            if auto_start:
                background_tasks.add_task(
                    self._run_training, db_job.id, job.parameters.dict()
                )

            logger.info(
                "Training job created",
                job_id=db_job.id,
                model_type=job.model_type,
                auto_start=auto_start,
            )

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
        model_type: Optional[str] = None,
        created_by: Optional[str] = None,
        search: Optional[str] = None,
        order_by: str = "created_at",
        order_desc: bool = True,
    ) -> TrainingJobList:
        """Get training jobs list"""
        try:
            query = self.db.query(TrainingJob).filter(TrainingJob.del_flag == 0)

            if status:
                query = query.filter(TrainingJob.status == status)

            if model_type:
                query = query.filter(TrainingJob.model_type == model_type)

            # created_by is accepted for API compatibility, but not persisted yet
            if created_by:
                logger.warning(
                    "created_by filter ignored: field not implemented",
                    created_by=created_by,
                )

            if search:
                search_term = f"%{search}%"
                query = query.filter(
                    or_(
                        TrainingJob.name.ilike(search_term),
                        TrainingJob.description.ilike(search_term),
                    )
                )

            if hasattr(TrainingJob, order_by):
                order_column = getattr(TrainingJob, order_by)
                query = query.order_by(
                    order_column.desc() if order_desc else order_column.asc()
                )
            else:
                query = query.order_by(TrainingJob.created_at.desc())

            total = query.count()
            results = query.offset(skip).limit(limit).all()

            jobs = [self._db_to_response(job) for job in results]

            return TrainingJobList(
                jobs=jobs,
                total=total,
                page=skip // limit + 1,
                size=limit,
                pages=(total + limit - 1) // limit,
            )

        except Exception as e:
            logger.error("Failed to get training jobs", error=str(e))
            raise

    async def get_job(self, job_id: int) -> Optional[TrainingJobResponse]:
        """Get training job by ID"""
        try:
            job = (
                self.db.query(TrainingJob)
                .filter(TrainingJob.id == job_id, TrainingJob.del_flag == 0)
                .first()
            )

            if not job:
                return None

            return self._db_to_response(job)

        except Exception as e:
            logger.error("Failed to get training job", error=str(e), job_id=job_id)
            raise

    async def update_job(
        self, job_id: int, job_update: TrainingJobUpdate
    ) -> Optional[TrainingJobResponse]:
        """Update training job"""
        try:
            job = (
                self.db.query(TrainingJob)
                .filter(TrainingJob.id == job_id, TrainingJob.del_flag == 0)
                .first()
            )

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
            job = (
                self.db.query(TrainingJob)
                .filter(TrainingJob.id == job_id, TrainingJob.del_flag == 0)
                .first()
            )

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
            job = (
                self.db.query(TrainingJob)
                .filter(TrainingJob.id == job_id, TrainingJob.del_flag == 0)
                .first()
            )

            if not job:
                return None

            active_meta = self._active_jobs.get(job_id, {})

            current_epoch = active_meta.get("current_epoch")
            current_loss = active_meta.get("current_loss")
            current_accuracy = active_meta.get("current_accuracy")
            message = active_meta.get("message")

            if current_epoch is None and job.status in [
                JobStatus.COMPLETED.value,
                JobStatus.FAILED.value,
                JobStatus.CANCELLED.value,
            ]:
                current_epoch = job.epochs

            if current_loss is None:
                current_loss = job.loss

            if current_accuracy is None:
                current_accuracy = job.accuracy

            return TrainingProgress(
                job_id=job.id,
                status=self._normalize_job_status(job.status),
                progress=job.progress,
                current_epoch=current_epoch,
                total_epochs=job.epochs,
                current_loss=current_loss,
                current_accuracy=current_accuracy,
                estimated_time_remaining=None,  # TODO: Calculate ETA
                message=message,
            )

        except Exception as e:
            logger.error("Failed to get training progress", error=str(e), job_id=job_id)
            raise

    async def get_metrics(self) -> TrainingMetrics:
        """Get training metrics"""
        try:
            # Get job counts
            total_jobs = (
                self.db.query(TrainingJob).filter(TrainingJob.del_flag == 0).count()
            )
            active_jobs = (
                self.db.query(TrainingJob)
                .filter(
                    TrainingJob.status.in_(
                        [JobStatus.PENDING.value, JobStatus.RUNNING.value]
                    ),
                    TrainingJob.del_flag == 0,
                )
                .count()
            )
            completed_jobs = (
                self.db.query(TrainingJob)
                .filter(
                    TrainingJob.status == JobStatus.COMPLETED.value,
                    TrainingJob.del_flag == 0,
                )
                .count()
            )
            failed_jobs = (
                self.db.query(TrainingJob)
                .filter(
                    TrainingJob.status == JobStatus.FAILED.value,
                    TrainingJob.del_flag == 0,
                )
                .count()
            )

            # Calculate average accuracy
            completed_with_accuracy = (
                self.db.query(TrainingJob.accuracy)
                .filter(
                    TrainingJob.status == JobStatus.COMPLETED.value,
                    TrainingJob.accuracy.isnot(None),
                    TrainingJob.del_flag == 0,
                )
                .all()
            )
            average_accuracy = (
                sum(acc[0] for acc in completed_with_accuracy)
                / len(completed_with_accuracy)
                if completed_with_accuracy
                else None
            )

            # Calculate average training time
            completed_with_time = (
                self.db.query(TrainingJob.started_at, TrainingJob.completed_at)
                .filter(
                    TrainingJob.status == JobStatus.COMPLETED.value,
                    TrainingJob.started_at.isnot(None),
                    TrainingJob.completed_at.isnot(None),
                    TrainingJob.del_flag == 0,
                )
                .all()
            )

            training_times = []
            for started, completed in completed_with_time:
                if started and completed:
                    training_times.append((completed - started).total_seconds())

            average_training_time = (
                sum(training_times) / len(training_times) if training_times else None
            )

            # Jobs by model type
            jobs_by_model_type = {}
            for model_type in settings.SUPPORTED_MODELS:
                count = (
                    self.db.query(TrainingJob)
                    .filter(
                        TrainingJob.model_type == model_type, TrainingJob.del_flag == 0
                    )
                    .count()
                )
                if count > 0:
                    jobs_by_model_type[model_type] = count

            # Jobs by status
            jobs_by_status = {
                "pending": self.db.query(TrainingJob)
                .filter(
                    TrainingJob.status == JobStatus.PENDING.value,
                    TrainingJob.del_flag == 0,
                )
                .count(),
                "running": self.db.query(TrainingJob)
                .filter(
                    TrainingJob.status == JobStatus.RUNNING.value,
                    TrainingJob.del_flag == 0,
                )
                .count(),
                "completed": completed_jobs,
                "failed": failed_jobs,
                "cancelled": self.db.query(TrainingJob)
                .filter(
                    TrainingJob.status == JobStatus.CANCELLED.value,
                    TrainingJob.del_flag == 0,
                )
                .count(),
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
                jobs_by_status=jobs_by_status,
            )

        except Exception as e:
            logger.error("Failed to get training metrics", error=str(e))
            raise

    async def start_job(
        self, job_id: int, background_tasks: BackgroundTasks, current_user: str
    ) -> bool:
        """Start a pending training job manually"""
        try:
            job = (
                self.db.query(TrainingJob)
                .filter(TrainingJob.id == job_id, TrainingJob.del_flag == 0)
                .first()
            )

            if not job:
                return False

            if job.status != JobStatus.PENDING.value:
                return False

            job.error_message = None
            self.db.commit()

            background_tasks.add_task(
                self._run_training,
                job_id,
                {
                    "epochs": job.epochs,
                    "learning_rate": job.learning_rate,
                    "batch_size": job.batch_size,
                },
            )

            logger.info(
                "Training job start scheduled", job_id=job_id, user=current_user
            )
            return True
        except Exception as e:
            logger.error(
                "Failed to schedule training job",
                error=str(e),
                job_id=job_id,
                user=current_user,
            )
            self.db.rollback()
            raise

    async def stop_job(self, job_id: int, current_user: str) -> bool:
        """Stop a pending or running training job"""
        try:
            job = (
                self.db.query(TrainingJob)
                .filter(TrainingJob.id == job_id, TrainingJob.del_flag == 0)
                .first()
            )

            if not job:
                return False

            if job.status not in [JobStatus.PENDING.value, JobStatus.RUNNING.value]:
                return False

            job.status = JobStatus.CANCELLED.value
            job.completed_at = datetime.now()
            self.db.commit()

            logger.info("Training job cancelled", job_id=job_id, user=current_user)
            return True
        except Exception as e:
            logger.error(
                "Failed to stop training job",
                error=str(e),
                job_id=job_id,
                user=current_user,
            )
            self.db.rollback()
            raise

    async def get_job_logs(self, job_id: int, tail_lines: int = 100) -> List[str]:
        """Get lightweight job logs from job state"""
        try:
            job = (
                self.db.query(TrainingJob)
                .filter(TrainingJob.id == job_id, TrainingJob.del_flag == 0)
                .first()
            )

            if not job:
                return []

            lines = [
                f"[{job.created_at}] Job created: {job.name} (type={job.model_type})",
                f"Status: {job.status}",
                f"Progress: {job.progress:.2f}%",
            ]

            if job.started_at:
                lines.append(f"[{job.started_at}] Training started")
            if job.completed_at:
                lines.append(
                    f"[{job.completed_at}] Training finished with status={job.status}"
                )
            if job.accuracy is not None:
                lines.append(f"Accuracy: {job.accuracy:.4f}")
            if job.loss is not None:
                lines.append(f"Loss: {job.loss:.4f}")
            if job.error_message:
                lines.append(f"Error: {job.error_message}")

            return lines[-tail_lines:] if tail_lines > 0 else lines
        except Exception as e:
            logger.error("Failed to get training job logs", error=str(e), job_id=job_id)
            raise

    async def retain_model_file(self, job_id: int, current_user: str) -> Optional[str]:
        """Confirm retaining trained model file for a completed job."""
        try:
            job = (
                self.db.query(TrainingJob)
                .filter(TrainingJob.id == job_id, TrainingJob.del_flag == 0)
                .first()
            )

            if not job:
                return None

            if job.status != JobStatus.COMPLETED.value:
                raise ValueError("Only completed training jobs can retain model files")

            if not job.model_path:
                raise ValueError("No model file available for this job")

            model_path = Path(job.model_path).expanduser()
            if not model_path.is_absolute():
                model_path = (Path.cwd() / model_path).resolve()

            if not model_path.exists() or not model_path.is_file():
                raise ValueError("Model file does not exist on disk")

            logger.info(
                "Model file retention confirmed",
                job_id=job_id,
                user=current_user,
                model_path=str(model_path),
            )
            return str(model_path)
        except Exception as e:
            logger.error(
                "Failed to retain model file",
                error=str(e),
                job_id=job_id,
                user=current_user,
            )
            raise

    async def discard_model_file(self, job_id: int, current_user: str) -> bool:
        """Delete trained model file for a completed job and clear model path."""
        try:
            job = (
                self.db.query(TrainingJob)
                .filter(TrainingJob.id == job_id, TrainingJob.del_flag == 0)
                .first()
            )

            if not job:
                return False

            if job.status != JobStatus.COMPLETED.value:
                raise ValueError("Only completed training jobs can discard model files")

            if not job.model_path:
                return True

            model_path = Path(job.model_path).expanduser()
            if not model_path.is_absolute():
                model_path = (Path.cwd() / model_path).resolve()

            if model_path.exists() and model_path.is_file():
                model_path.unlink()

            job.model_path = None
            self.db.commit()

            logger.info(
                "Model file discarded",
                job_id=job_id,
                user=current_user,
                model_path=str(model_path),
            )
            return True
        except Exception as e:
            logger.error(
                "Failed to discard model file",
                error=str(e),
                job_id=job_id,
                user=current_user,
            )
            self.db.rollback()
            raise

    async def get_job_statistics(self) -> Dict[str, Any]:
        """Get training job statistics overview"""
        try:
            base_query = self.db.query(TrainingJob).filter(TrainingJob.del_flag == 0)
            total_jobs = base_query.count()

            by_status = {
                status.value: self.db.query(TrainingJob)
                .filter(TrainingJob.del_flag == 0, TrainingJob.status == status.value)
                .count()
                for status in JobStatus
            }

            by_model_type: Dict[str, int] = {}
            for model_type in settings.SUPPORTED_MODELS:
                count = (
                    self.db.query(TrainingJob)
                    .filter(
                        TrainingJob.del_flag == 0, TrainingJob.model_type == model_type
                    )
                    .count()
                )
                if count > 0:
                    by_model_type[model_type] = count

            avg_progress_rows = base_query.with_entities(TrainingJob.progress).all()
            average_progress = (
                sum(row[0] for row in avg_progress_rows if row[0] is not None)
                / len(avg_progress_rows)
                if avg_progress_rows
                else 0.0
            )

            return {
                "total_jobs": total_jobs,
                "jobs_by_status": by_status,
                "jobs_by_model_type": by_model_type,
                "average_progress": round(average_progress, 2),
            }
        except Exception as e:
            logger.error("Failed to get training job statistics", error=str(e))
            raise

    # Private methods

    async def _run_training(self, job_id: int, parameters: Dict[str, Any]):
        """Run training job in background."""
        model_type = None
        try:
            with get_db_session() as db:
                job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
                if not job:
                    return

                if job.status == JobStatus.CANCELLED.value:
                    logger.info("Training skipped for cancelled job", job_id=job_id)
                    return

                model_type = job.model_type
                job.status = JobStatus.RUNNING.value
                job.started_at = datetime.now()
                job.progress = 0.0
                job.error_message = None

            mode = "sequence" if model_type == "lrcn" else "image"
            self._set_active_job_metadata(
                job_id,
                {
                    "status": JobStatus.RUNNING.value,
                    "mode": mode,
                    "total_epochs": int(parameters.get("epochs", 0)),
                    "current_epoch": 0,
                    "current_loss": None,
                    "current_accuracy": None,
                    "message": f"Starting {mode} training",
                },
            )

            trainer = (
                self._train_sequence_model
                if model_type == "lrcn"
                else self._train_image_model
            )
            await asyncio.to_thread(trainer, job_id, parameters)

        except TrainingCancelledError:
            logger.info("Training cancelled", job_id=job_id)

            with get_db_session() as db:
                job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
                if job and job.status != JobStatus.CANCELLED.value:
                    job.status = JobStatus.CANCELLED.value
                if job and not job.completed_at:
                    job.completed_at = datetime.now()

            self._set_active_job_metadata(
                job_id,
                {
                    "status": JobStatus.CANCELLED.value,
                    "message": "Training cancelled",
                },
            )

        except Exception as e:
            logger.error("Training failed", error=str(e), job_id=job_id)

            with get_db_session() as db:
                job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
                if job:
                    job.status = JobStatus.FAILED.value
                    job.error_message = str(e)
                    job.completed_at = datetime.now()

            self._set_active_job_metadata(
                job_id,
                {
                    "status": JobStatus.FAILED.value,
                    "message": f"Training failed: {str(e)}",
                },
            )

    def _train_image_model(self, job_id: int, parameters: Dict[str, Any]):
        """Train an image classification model and persist epoch progress."""
        with get_db_session() as db:
            job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if not job:
                raise ValueError(f"Training job {job_id} not found")

            dataset_path = job.dataset_path
            model_type = job.model_type
            epochs = int(parameters.get("epochs", job.epochs))
            learning_rate = float(parameters.get("learning_rate", job.learning_rate))
            batch_size = int(parameters.get("batch_size", job.batch_size))
            validation_split = float(parameters.get("validation_split", 0.2))

        self._set_active_job_metadata(job_id, {"total_epochs": epochs, "mode": "image"})

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader, val_loader = self._build_dataloaders(
            dataset_path, batch_size, validation_split
        )

        model = create_model(model_type=model_type, num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            trainable_params if trainable_params else model.parameters(),
            lr=learning_rate,
        )

        model_dir = Path(settings.MODEL_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)
        best_checkpoint_path = model_dir / f"training_job_{job_id}_best.pth"

        best_val_accuracy = -1.0
        best_val_loss = None

        for epoch in range(1, epochs + 1):
            if self._is_job_cancelled(job_id):
                raise TrainingCancelledError(
                    f"Training job {job_id} cancelled at epoch {epoch}"
                )

            train_loss, train_accuracy = self._run_train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_accuracy = self._run_eval_epoch(
                model, val_loader, criterion, device
            )

            progress = (epoch / epochs) * 100.0
            self._update_epoch_metrics(
                job_id,
                progress,
                val_accuracy,
                val_loss,
                epoch=epoch,
                total_epochs=epochs,
                message=f"Epoch {epoch}/{epochs} completed",
            )

            logger.info(
                "Training epoch completed",
                job_id=job_id,
                epoch=epoch,
                total_epochs=epochs,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                progress=progress,
            )

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_val_loss = val_loss
                torch.save(
                    {
                        "job_id": job_id,
                        "model_type": model_type,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_accuracy": val_accuracy,
                        "val_loss": val_loss,
                        "input_size": settings.MODEL_INPUT_SIZE,
                        "saved_at": datetime.now().isoformat(),
                    },
                    str(best_checkpoint_path),
                )

        self._finalize_successful_training(
            job_id,
            best_val_accuracy,
            best_val_loss,
            best_checkpoint_path,
        )

    def _train_sequence_model(self, job_id: int, parameters: Dict[str, Any]):
        """Train the LRCN model using frame-sequence clips."""
        with get_db_session() as db:
            job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if not job:
                raise ValueError(f"Training job {job_id} not found")

            dataset_path = job.dataset_path
            model_type = job.model_type
            epochs = int(parameters.get("epochs", job.epochs))
            learning_rate = float(parameters.get("learning_rate", job.learning_rate))
            batch_size = int(parameters.get("batch_size", job.batch_size))
            validation_split = float(parameters.get("validation_split", 0.2))
            sequence_length = int(parameters.get("sequence_length", 16))
            frame_stride = int(parameters.get("frame_stride", 1))
            hidden_size = int(parameters.get("hidden_size", 512))
            num_layers = int(parameters.get("num_layers", 2))

        if sequence_length <= 0:
            raise ValueError("sequence_length must be greater than 0")
        if frame_stride <= 0:
            raise ValueError("frame_stride must be greater than 0")

        self._set_active_job_metadata(
            job_id,
            {
                "total_epochs": epochs,
                "mode": "sequence",
                "message": f"Preparing clips (sequence_length={sequence_length}, frame_stride={frame_stride})",
            },
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader, val_loader = self._build_sequence_dataloaders(
            dataset_path, batch_size, validation_split, sequence_length, frame_stride
        )

        model = create_model(
            model_type=model_type,
            num_classes=2,
            input_size=25088,
            hidden_size=hidden_size,
            num_layers=num_layers,
        ).to(device)
        criterion = nn.CrossEntropyLoss()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            trainable_params if trainable_params else model.parameters(),
            lr=learning_rate,
        )

        model_dir = Path(settings.MODEL_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)
        best_checkpoint_path = model_dir / f"training_job_{job_id}_best.pth"

        best_val_accuracy = -1.0
        best_val_loss = None

        for epoch in range(1, epochs + 1):
            if self._is_job_cancelled(job_id):
                raise TrainingCancelledError(
                    f"Training job {job_id} cancelled at epoch {epoch}"
                )

            train_loss, train_accuracy = self._run_train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_accuracy = self._run_eval_epoch(
                model, val_loader, criterion, device
            )

            progress = (epoch / epochs) * 100.0
            self._update_epoch_metrics(
                job_id,
                progress,
                val_accuracy,
                val_loss,
                epoch=epoch,
                total_epochs=epochs,
                message=f"Sequence epoch {epoch}/{epochs} completed",
            )

            logger.info(
                "Sequence training epoch completed",
                job_id=job_id,
                epoch=epoch,
                total_epochs=epochs,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                progress=progress,
                sequence_length=sequence_length,
                frame_stride=frame_stride,
            )

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_val_loss = val_loss
                torch.save(
                    {
                        "job_id": job_id,
                        "model_type": model_type,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_accuracy": val_accuracy,
                        "val_loss": val_loss,
                        "input_size": settings.MODEL_INPUT_SIZE,
                        "sequence_length": sequence_length,
                        "frame_stride": frame_stride,
                        "saved_at": datetime.now().isoformat(),
                    },
                    str(best_checkpoint_path),
                )

        self._finalize_successful_training(
            job_id,
            best_val_accuracy,
            best_val_loss,
            best_checkpoint_path,
        )

    def _finalize_successful_training(
        self,
        job_id: int,
        best_val_accuracy: float,
        best_val_loss: Optional[float],
        best_checkpoint_path: Path,
    ):
        """Persist final job state after successful training."""
        with get_db_session() as db:
            job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if not job:
                return
            if job.status == JobStatus.CANCELLED.value:
                job.completed_at = job.completed_at or datetime.now()
                return

            job.status = JobStatus.COMPLETED.value
            job.progress = 100.0
            job.completed_at = datetime.now()
            job.accuracy = best_val_accuracy if best_val_accuracy >= 0 else None
            job.loss = best_val_loss
            job.model_path = str(best_checkpoint_path)

        self._set_active_job_metadata(
            job_id,
            {
                "status": JobStatus.COMPLETED.value,
                "current_epoch": None,
                "current_loss": best_val_loss,
                "current_accuracy": best_val_accuracy
                if best_val_accuracy >= 0
                else None,
                "message": "Training completed",
            },
        )

    def _build_dataloaders(
        self, dataset_path: str, batch_size: int, validation_split: float
    ) -> Tuple[DataLoader, DataLoader]:
        """Build train/validation dataloaders from image dataset path."""
        root_path = Path(dataset_path).expanduser()
        if not root_path.exists() or not root_path.is_dir():
            raise ValueError(
                f"Dataset path does not exist or is not a directory: {dataset_path}"
            )

        input_size = int(settings.MODEL_INPUT_SIZE)
        train_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        split_dirs = self._find_split_dirs(root_path)
        if split_dirs:
            train_samples = self._collect_labeled_images(split_dirs["train"], root_path)
            val_samples = self._collect_labeled_images(split_dirs["val"], root_path)
        else:
            all_samples = self._collect_labeled_images(root_path, root_path)
            if len(all_samples) < 2:
                raise ValueError(
                    "Dataset must contain at least 2 labeled images for train/validation split"
                )

            validation_split = max(0.0, min(0.9, validation_split))
            val_size = max(1, int(len(all_samples) * validation_split))
            train_size = len(all_samples) - val_size
            if train_size <= 0:
                train_size = len(all_samples) - 1
                val_size = 1

            shuffled = all_samples[:]
            generator = torch.Generator().manual_seed(42)
            indices = torch.randperm(len(shuffled), generator=generator).tolist()
            shuffled = [shuffled[idx] for idx in indices]

            train_samples = shuffled[:train_size]
            val_samples = shuffled[train_size:]

        if not train_samples:
            raise ValueError("No labeled training images found")
        if not val_samples:
            raise ValueError("No labeled validation images found")

        train_dataset = ImageClassificationDataset(
            train_samples, transform=train_transform
        )
        val_dataset = ImageClassificationDataset(val_samples, transform=val_transform)

        pin_memory = torch.cuda.is_available()
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )

        return train_loader, val_loader

    def _build_sequence_dataloaders(
        self,
        dataset_path: str,
        batch_size: int,
        validation_split: float,
        sequence_length: int,
        frame_stride: int,
    ) -> Tuple[DataLoader, DataLoader]:
        """Build train/validation dataloaders for frame sequence clips."""
        root_path = Path(dataset_path).expanduser()
        if not root_path.exists() or not root_path.is_dir():
            raise ValueError(
                f"Dataset path does not exist or is not a directory: {dataset_path}"
            )

        input_size = int(settings.MODEL_INPUT_SIZE)
        train_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        split_dirs = self._find_split_dirs(root_path)
        if split_dirs:
            train_videos = self._collect_labeled_video_groups(
                split_dirs["train"], split_dirs["train"]
            )
            val_videos = self._collect_labeled_video_groups(
                split_dirs["val"], split_dirs["val"]
            )
        else:
            all_videos = self._collect_labeled_video_groups(root_path, root_path)
            if len(all_videos) < 2:
                raise ValueError(
                    "Dataset must contain at least 2 labeled video groups for train/validation split"
                )

            validation_split = max(0.0, min(0.9, validation_split))
            val_size = max(1, int(len(all_videos) * validation_split))
            train_size = len(all_videos) - val_size
            if train_size <= 0:
                train_size = len(all_videos) - 1
                val_size = 1

            generator = torch.Generator().manual_seed(42)
            indices = torch.randperm(len(all_videos), generator=generator).tolist()
            shuffled = [all_videos[idx] for idx in indices]
            train_videos = shuffled[:train_size]
            val_videos = shuffled[train_size:]

        train_samples: List[Tuple[List[str], int]] = []
        for frame_paths, label in train_videos:
            train_samples.extend(
                self._build_clips_from_video_frames(
                    frame_paths, label, sequence_length, frame_stride
                )
            )

        val_samples: List[Tuple[List[str], int]] = []
        for frame_paths, label in val_videos:
            val_samples.extend(
                self._build_clips_from_video_frames(
                    frame_paths, label, sequence_length, frame_stride
                )
            )

        if not train_samples:
            raise ValueError("No labeled training clips found")
        if not val_samples:
            raise ValueError("No labeled validation clips found")

        train_dataset = SequenceClassificationDataset(
            train_samples, transform=train_transform
        )
        val_dataset = SequenceClassificationDataset(
            val_samples, transform=val_transform
        )

        pin_memory = torch.cuda.is_available()
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )

        return train_loader, val_loader

    def _collect_labeled_video_groups(
        self, root_path: Path, label_root: Path
    ) -> List[Tuple[List[str], int]]:
        """Collect frame paths grouped by video directory and label."""
        grouped_frames: Dict[str, Dict[str, Any]] = {}

        for file_path in root_path.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue

            label = self._infer_label_from_path(file_path, label_root)
            if label is None:
                continue

            if not self._is_valid_image(file_path):
                logger.warning("Skipping invalid image file", path=str(file_path))
                continue

            group_key = str(file_path.parent)
            if group_key not in grouped_frames:
                grouped_frames[group_key] = {"label": label, "frames": []}
            grouped_frames[group_key]["frames"].append(str(file_path))

        videos: List[Tuple[List[str], int]] = []
        for group_data in grouped_frames.values():
            sorted_frames = sorted(group_data["frames"], key=self._natural_sort_key)
            if sorted_frames:
                videos.append((sorted_frames, group_data["label"]))

        return videos

    def _build_clips_from_video_frames(
        self,
        frame_paths: List[str],
        label: int,
        sequence_length: int,
        frame_stride: int,
    ) -> List[Tuple[List[str], int]]:
        """Create fixed-length clips from sorted frame paths."""
        if not frame_paths:
            return []

        clip_span = ((sequence_length - 1) * frame_stride) + 1
        clips: List[Tuple[List[str], int]] = []

        if len(frame_paths) < sequence_length:
            padded = frame_paths + [frame_paths[-1]] * (
                sequence_length - len(frame_paths)
            )
            return [(padded, label)]

        if len(frame_paths) < clip_span:
            clip = [
                frame_paths[min(i * frame_stride, len(frame_paths) - 1)]
                for i in range(sequence_length)
            ]
            return [(clip, label)]

        step = max(1, clip_span)
        for start in range(0, len(frame_paths), step):
            clip = []
            for i in range(sequence_length):
                idx = start + (i * frame_stride)
                if idx >= len(frame_paths):
                    idx = len(frame_paths) - 1
                clip.append(frame_paths[idx])
            clips.append((clip, label))

            if start + clip_span >= len(frame_paths):
                break

        return clips

    def _natural_sort_key(self, file_path: str) -> List[Any]:
        """Sort helper that keeps numeric frame indexes in order."""
        name = Path(file_path).name.lower()
        return [
            int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)
        ]

    def _find_split_dirs(self, root_path: Path) -> Optional[Dict[str, Path]]:
        """Detect explicit train/val split directories."""
        child_dirs = {
            child.name.lower(): child for child in root_path.iterdir() if child.is_dir()
        }

        train_dir = child_dirs.get("train")
        val_dir = (
            child_dirs.get("val")
            or child_dirs.get("valid")
            or child_dirs.get("validation")
        )

        if train_dir and val_dir:
            return {"train": train_dir, "val": val_dir}
        return None

    def _collect_labeled_images(
        self, root_path: Path, label_root: Path
    ) -> List[Tuple[str, int]]:
        """Collect valid image files and infer labels from directory names."""
        samples: List[Tuple[str, int]] = []

        for file_path in root_path.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue

            label = self._infer_label_from_path(file_path, label_root)
            if label is None:
                continue

            if not self._is_valid_image(file_path):
                logger.warning("Skipping invalid image file", path=str(file_path))
                continue

            samples.append((str(file_path), label))

        return samples

    def _infer_label_from_path(self, file_path: Path, root_path: Path) -> Optional[int]:
        """Infer class label from parent directory keywords."""
        try:
            relative_parents = list(file_path.relative_to(root_path).parents)
        except ValueError:
            relative_parents = list(file_path.parents)

        for parent in relative_parents:
            if str(parent) in {"", "."}:
                continue

            name = parent.name.lower()
            is_fake = self._matches_keyword(name, FAKE_LABEL_KEYWORDS)
            is_real = self._matches_keyword(name, REAL_LABEL_KEYWORDS)

            if is_fake and not is_real:
                return 0
            if is_real and not is_fake:
                return 1

        return None

    def _matches_keyword(self, value: str, keywords: set) -> bool:
        """Check if a directory name matches any label keyword."""
        normalized = "".join(ch if ch.isalnum() else " " for ch in value.lower())
        tokens = [token for token in normalized.split() if token]

        for keyword in keywords:
            if keyword in tokens:
                return True
            if keyword in normalized:
                return True
        return False

    def _is_valid_image(self, path: Path) -> bool:
        """Return True if PIL can open and verify the image."""
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except (UnidentifiedImageError, OSError, ValueError):
            return False

    def _run_train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Tuple[float, float]:
        """Run one training epoch."""
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            current_batch_size = labels.size(0)
            total_loss += loss.item() * current_batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += current_batch_size

        average_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        return average_loss, accuracy

    def _run_eval_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> Tuple[float, float]:
        """Run one validation epoch."""
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                current_batch_size = labels.size(0)
                total_loss += loss.item() * current_batch_size
                total_correct += (outputs.argmax(dim=1) == labels).sum().item()
                total_samples += current_batch_size

        average_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        return average_loss, accuracy

    def _update_epoch_metrics(
        self,
        job_id: int,
        progress: float,
        accuracy: float,
        loss: float,
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
        message: Optional[str] = None,
    ):
        """Persist epoch metrics to database and active metadata."""
        clamped_progress = min(max(progress, 0.0), 100.0)
        with get_db_session() as db:
            job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if not job:
                return

            if job.status == JobStatus.CANCELLED.value:
                raise TrainingCancelledError(f"Training job {job_id} cancelled")

            job.progress = clamped_progress
            job.accuracy = accuracy
            job.loss = loss

        active_update: Dict[str, Any] = {
            "progress": clamped_progress,
            "current_loss": loss,
            "current_accuracy": accuracy,
        }
        if epoch is not None:
            active_update["current_epoch"] = epoch
        if total_epochs is not None:
            active_update["total_epochs"] = total_epochs
        if message is not None:
            active_update["message"] = message
        self._set_active_job_metadata(job_id, active_update)

    def _set_active_job_metadata(self, job_id: int, updates: Dict[str, Any]):
        """Merge metadata updates for a running training job."""
        if job_id not in self._active_jobs:
            self._active_jobs[job_id] = {}
        self._active_jobs[job_id].update(updates)

    def _is_job_cancelled(self, job_id: int) -> bool:
        """Check whether job status has been set to cancelled."""
        with get_db_session() as db:
            job = db.query(TrainingJob.status).filter(TrainingJob.id == job_id).first()
            if not job:
                return True
            return job[0] == JobStatus.CANCELLED.value

    async def _cancel_training(self, job_id: int):
        """Cancel running training job"""
        with get_db_session() as db:
            job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if job and job.status in [JobStatus.PENDING.value, JobStatus.RUNNING.value]:
                job.status = JobStatus.CANCELLED.value
                if not job.completed_at:
                    job.completed_at = datetime.now()

        if job_id in self._active_jobs:
            del self._active_jobs[job_id]

    def _db_to_response(self, job: TrainingJob) -> TrainingJobResponse:
        """Convert database model to response schema"""
        from app.schemas.training import TrainingParameters, TrainingResults

        parameters = TrainingParameters(
            epochs=job.epochs if job.epochs is not None else settings.DEFAULT_EPOCHS,
            learning_rate=job.learning_rate
            if job.learning_rate is not None
            else settings.DEFAULT_LEARNING_RATE,
            batch_size=job.batch_size
            if job.batch_size is not None
            else settings.MODEL_BATCH_SIZE,
        )

        results = None
        if job.accuracy is not None:
            results = TrainingResults(
                accuracy=job.accuracy, loss=job.loss, model_path=job.model_path
            )

        return TrainingJobResponse(
            id=job.id,
            name=job.name,
            description=job.description,
            model_type=job.model_type,
            dataset_path=job.dataset_path,
            parameters=parameters,
            status=self._normalize_job_status(job.status),
            progress=job.progress if job.progress is not None else 0.0,
            results=results,
            created_at=job.created_at,
            updated_at=job.updated_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
        )

    def _normalize_job_status(self, raw_status: Optional[str]) -> JobStatus:
        """Normalize persisted status values to supported enum values."""
        status_value = raw_status or JobStatus.PENDING.value
        try:
            return JobStatus(status_value)
        except Exception:
            return JobStatus.PENDING
