"""
Dataset management service for deepfake detection platform
"""

import os
import time
import asyncio
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session

from app.core.logging import logger
from app.core.config import settings
from app.models.database_models import DatasetInfo
from app.schemas.datasets import (
    DatasetCreate, DatasetResponse, DatasetList, DatasetUpdate,
    DatasetStats, DatasetProcessingConfig
)


class DatasetService:
    """Service for managing datasets and data processing"""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def create_dataset(self, dataset: DatasetCreate) -> DatasetResponse:
        """Create a new dataset registry entry"""
        try:
            # Check if dataset name already exists
            existing = self.db.query(DatasetInfo).filter(
                DatasetInfo.name == dataset.name,
                DatasetInfo.del_flag == 0
            ).first()
            
            if existing:
                raise ValueError(f"Dataset with name '{dataset.name}' already exists")
            
            # Create dataset entry
            db_dataset = DatasetInfo(
                name=dataset.name,
                description=dataset.description,
                path=dataset.path,
                image_size=dataset.image_size,
                frame_extraction_interval=dataset.frame_extraction_interval,
                max_frames_per_video=dataset.max_frames_per_video
            )
            
            self.db.add(db_dataset)
            self.db.commit()
            self.db.refresh(db_dataset)
            
            logger.info("Dataset created", dataset_name=dataset.name, path=dataset.path)
            
            return self._db_to_response(db_dataset)
            
        except Exception as e:
            logger.error("Failed to create dataset", error=str(e))
            self.db.rollback()
            raise
    
    async def get_datasets(
        self,
        skip: int = 0,
        limit: int = 100,
        is_processed: Optional[bool] = None
    ) -> DatasetList:
        """Get datasets list"""
        try:
            query = self.db.query(DatasetInfo).filter(DatasetInfo.del_flag == 0)
            
            if is_processed is not None:
                query = query.filter(DatasetInfo.is_processed == is_processed)
            
            total = query.count()
            results = query.order_by(DatasetInfo.created_at.desc()).offset(skip).limit(limit).all()
            
            datasets = [self._db_to_response(dataset) for dataset in results]
            
            return DatasetList(
                datasets=datasets,
                total=total,
                page=skip // limit + 1,
                size=limit,
                pages=(total + limit - 1) // limit
            )
            
        except Exception as e:
            logger.error("Failed to get datasets", error=str(e))
            raise
    
    async def get_dataset(self, dataset_id: int) -> Optional[DatasetResponse]:
        """Get dataset by ID"""
        try:
            dataset = self.db.query(DatasetInfo).filter(
                DatasetInfo.id == dataset_id,
                DatasetInfo.del_flag == 0
            ).first()
            
            if not dataset:
                return None
            
            return self._db_to_response(dataset)
            
        except Exception as e:
            logger.error("Failed to get dataset", error=str(e), dataset_id=dataset_id)
            raise
    
    async def update_dataset(self, dataset_id: int, dataset_update: DatasetUpdate) -> Optional[DatasetResponse]:
        """Update dataset"""
        try:
            dataset = self.db.query(DatasetInfo).filter(
                DatasetInfo.id == dataset_id,
                DatasetInfo.del_flag == 0
            ).first()
            
            if not dataset:
                return None
            
            # Update fields
            update_data = dataset_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                if hasattr(dataset, field):
                    setattr(dataset, field, value)
            
            self.db.commit()
            self.db.refresh(dataset)
            
            logger.info("Dataset updated", dataset_id=dataset_id)
            
            return self._db_to_response(dataset)
            
        except Exception as e:
            logger.error("Failed to update dataset", error=str(e), dataset_id=dataset_id)
            self.db.rollback()
            raise
    
    async def delete_dataset(self, dataset_id: int) -> bool:
        """Delete dataset"""
        try:
            dataset = self.db.query(DatasetInfo).filter(
                DatasetInfo.id == dataset_id,
                DatasetInfo.del_flag == 0
            ).first()
            
            if not dataset:
                return False
            
            dataset.del_flag = 1
            self.db.commit()
            
            logger.info("Dataset deleted", dataset_id=dataset_id)
            return True
            
        except Exception as e:
            logger.error("Failed to delete dataset", error=str(e), dataset_id=dataset_id)
            self.db.rollback()
            raise
    
    async def process_dataset(self, dataset_id: int, config: Dict[str, Any]):
        """Process dataset in background"""
        try:
            dataset = self.db.query(DatasetInfo).filter(
                DatasetInfo.id == dataset_id,
                DatasetInfo.del_flag == 0
            ).first()
            
            if not dataset:
                logger.error("Dataset not found for processing", dataset_id=dataset_id)
                return
            
            # Update status
            dataset.processing_status = "processing"
            self.db.commit()
            
            # Simulate dataset processing
            await self._simulate_processing(dataset_id, config)
            
        except Exception as e:
            logger.error("Dataset processing failed", error=str(e), dataset_id=dataset_id)
            
            # Update status to failed
            dataset = self.db.query(DatasetInfo).filter(DatasetInfo.id == dataset_id).first()
            if dataset:
                dataset.processing_status = "failed"
                dataset.error_message = str(e)
                self.db.commit()
    
    async def get_processing_status(self, dataset_id: int) -> Optional[Dict[str, Any]]:
        """Get dataset processing status"""
        try:
            dataset = self.db.query(DatasetInfo).filter(
                DatasetInfo.id == dataset_id,
                DatasetInfo.del_flag == 0
            ).first()
            
            if not dataset:
                return None
            
            return {
                "dataset_id": dataset_id,
                "processing_status": dataset.processing_status,
                "is_processed": dataset.is_processed,
                "error_message": dataset.error_message,
                "stats": self._get_dataset_stats(dataset) if dataset.is_processed else None
            }
            
        except Exception as e:
            logger.error("Failed to get dataset status", error=str(e), dataset_id=dataset_id)
            raise
    
    async def _simulate_processing(self, dataset_id: int, config: Dict[str, Any]):
        """Simulate dataset processing"""
        try:
            # Simulate processing time
            await asyncio.sleep(2)
            
            # Update dataset with simulated stats
            dataset = self.db.query(DatasetInfo).filter(DatasetInfo.id == dataset_id).first()
            if dataset:
                dataset.is_processed = True
                dataset.processing_status = "completed"
                dataset.total_samples = 1000
                dataset.real_samples = 500
                dataset.fake_samples = 500
                dataset.train_samples = 800
                dataset.val_samples = 100
                dataset.test_samples = 100
                self.db.commit()
            
            logger.info("Dataset processing completed", dataset_id=dataset_id)
            
        except Exception as e:
            logger.error("Dataset processing simulation failed", error=str(e), dataset_id=dataset_id)
            raise
    
    def _get_dataset_stats(self, dataset: DatasetInfo) -> DatasetStats:
        """Get dataset statistics"""
        return DatasetStats(
            total_samples=dataset.total_samples,
            real_samples=dataset.real_samples,
            fake_samples=dataset.fake_samples,
            train_samples=dataset.train_samples,
            val_samples=dataset.val_samples,
            test_samples=dataset.test_samples,
            class_distribution={
                "real": dataset.real_samples or 0,
                "fake": dataset.fake_samples or 0
            } if dataset.real_samples and dataset.fake_samples else None,
            data_quality_score=0.95  # Simulated quality score
        )
    
    def _db_to_response(self, dataset: DatasetInfo) -> DatasetResponse:
        """Convert database model to response schema"""
        stats = None
        if dataset.is_processed:
            stats = self._get_dataset_stats(dataset)
        
        return DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            path=dataset.path,
            image_size=dataset.image_size,
            frame_extraction_interval=dataset.frame_extraction_interval,
            max_frames_per_video=dataset.max_frames_per_video,
            stats=stats,
            is_processed=dataset.is_processed,
            processing_status=dataset.processing_status,
            error_message=dataset.error_message,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at
        )
