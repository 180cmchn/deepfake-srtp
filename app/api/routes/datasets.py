"""
Dataset management API routes for deepfake detection platform
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import uuid

from app.core.database import get_db
from app.core.logging import logger
from app.schemas.datasets import (
    DatasetCreate, DatasetResponse, DatasetList, DatasetUpdate,
    DatasetProcessingConfig
)
from app.services.dataset_service import DatasetService

router = APIRouter()


@router.post("/", response_model=DatasetResponse)
async def create_dataset(
    dataset: DatasetCreate,
    db: Session = Depends(get_db)
):
    """Create a new dataset registry entry"""
    try:
        dataset_service = DatasetService(db)
        result = await dataset_service.create_dataset(dataset)
        return result
    except Exception as e:
        logger.error("Failed to create dataset", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create dataset")


@router.get("/", response_model=DatasetList)
async def get_datasets(
    skip: int = 0,
    limit: int = 100,
    is_processed: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """Get datasets list"""
    try:
        dataset_service = DatasetService(db)
        result = await dataset_service.get_datasets(skip, limit, is_processed)
        return result
    except Exception as e:
        logger.error("Failed to get datasets", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get datasets")


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """Get dataset by ID"""
    try:
        dataset_service = DatasetService(db)
        result = await dataset_service.get_dataset(dataset_id)
        if not result:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get dataset", error=str(e), dataset_id=dataset_id)
        raise HTTPException(status_code=500, detail="Failed to get dataset")


@router.put("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: int,
    dataset_update: DatasetUpdate,
    db: Session = Depends(get_db)
):
    """Update dataset"""
    try:
        dataset_service = DatasetService(db)
        result = await dataset_service.update_dataset(dataset_id, dataset_update)
        if not result:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update dataset", error=str(e), dataset_id=dataset_id)
        raise HTTPException(status_code=500, detail="Failed to update dataset")


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """Delete dataset"""
    try:
        dataset_service = DatasetService(db)
        success = await dataset_service.delete_dataset(dataset_id)
        if not success:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return {"message": "Dataset deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete dataset", error=str(e), dataset_id=dataset_id)
        raise HTTPException(status_code=500, detail="Failed to delete dataset")


@router.post("/upload", response_model=DatasetResponse)
async def upload_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    name: str = None,
    description: str = None,
    config: DatasetProcessingConfig = Depends(),
    db: Session = Depends(get_db)
):
    """Upload and process dataset"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1].lower()
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join("data", unique_filename)
        
        # Save uploaded file
        os.makedirs("data", exist_ok=True)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create dataset entry
        dataset_data = DatasetCreate(
            name=name or file.filename,
            description=description,
            path=file_path,
            image_size=config.image_size,
            frame_extraction_interval=config.frame_extraction_interval,
            max_frames_per_video=config.max_frames_per_video
        )
        
        dataset_service = DatasetService(db)
        result = await dataset_service.create_dataset(dataset_data)
        
        # Start processing in background
        background_tasks.add_task(
            dataset_service.process_dataset,
            result.id,
            config.dict()
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to upload dataset", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to upload dataset")


@router.post("/{dataset_id}/process")
async def process_dataset(
    dataset_id: int,
    background_tasks: BackgroundTasks,
    config: DatasetProcessingConfig,
    db: Session = Depends(get_db)
):
    """Process dataset"""
    try:
        dataset_service = DatasetService(db)
        dataset = await dataset_service.get_dataset(dataset_id)
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Start processing in background
        background_tasks.add_task(
            dataset_service.process_dataset,
            dataset_id,
            config.dict()
        )
        
        return {"message": "Dataset processing started"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to process dataset", error=str(e), dataset_id=dataset_id)
        raise HTTPException(status_code=500, detail="Failed to process dataset")


@router.get("/{dataset_id}/status")
async def get_dataset_processing_status(dataset_id: int, db: Session = Depends(get_db)):
    """Get dataset processing status"""
    try:
        dataset_service = DatasetService(db)
        status = await dataset_service.get_processing_status(dataset_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get dataset status", error=str(e), dataset_id=dataset_id)
        raise HTTPException(status_code=500, detail="Failed to get dataset status")
