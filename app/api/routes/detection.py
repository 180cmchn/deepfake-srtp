"""
Detection API routes for deepfake detection platform
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import uuid
from datetime import datetime

from app.core.database import get_db
from app.core.logging import logger
from app.schemas.detection import (
    DetectionRequest, DetectionResponse, BatchDetectionRequest, 
    BatchDetectionResponse, VideoDetectionRequest, VideoDetectionResponse,
    DetectionHistory, DetectionHistoryList, DetectionStatistics
)
from app.services.detection_service import DetectionService

router = APIRouter()


@router.post("/detect", response_model=DetectionResponse)
async def detect_deepfake(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    request: DetectionRequest = Depends(),
    db: Session = Depends(get_db)
):
    """
    Detect deepfake in uploaded file
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1].lower()
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join("uploads", unique_filename)
        
        # Save uploaded file
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Initialize detection service
        detection_service = DetectionService(db)
        
        # Perform detection
        result = await detection_service.detect_file(
            file_path=file_path,
            request=request,
            background_tasks=background_tasks
        )
        
        logger.info("Detection completed", 
                   file_name=file.filename, 
                   result=result.prediction,
                   confidence=result.confidence)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Detection failed", error=str(e), file_name=file.filename)
        raise HTTPException(status_code=500, detail="Detection failed")


@router.post("/detect/batch", response_model=BatchDetectionResponse)
async def detect_deepfake_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    request: BatchDetectionRequest = Depends(),
    db: Session = Depends(get_db)
):
    """
    Detect deepfake in multiple files
    """
    try:
        if len(files) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 files allowed")
        
        # Save uploaded files
        file_paths = []
        for file in files:
            if not file.filename:
                continue
            
            file_extension = os.path.splitext(file.filename)[1].lower()
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = os.path.join("uploads", unique_filename)
            
            os.makedirs("uploads", exist_ok=True)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            file_paths.append(file_path)
        
        # Initialize detection service
        detection_service = DetectionService(db)
        
        # Perform batch detection
        result = await detection_service.detect_batch(
            file_paths=file_paths,
            request=request,
            background_tasks=background_tasks
        )
        
        logger.info("Batch detection completed", 
                   total_files=len(files),
                   processed_files=result.processed_files)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Batch detection failed", error=str(e))
        raise HTTPException(status_code=500, detail="Batch detection failed")


@router.post("/detect/video", response_model=VideoDetectionResponse)
async def detect_deepfake_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    request: VideoDetectionRequest = Depends(),
    db: Session = Depends(get_db)
):
    """
    Detect deepfake in video file
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Validate video file
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
            raise HTTPException(status_code=400, detail="Invalid video format")
        
        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join("uploads", unique_filename)
        
        # Save uploaded file
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Initialize detection service
        detection_service = DetectionService(db)
        
        # Perform video detection
        result = await detection_service.detect_video(
            video_path=file_path,
            request=request,
            background_tasks=background_tasks
        )
        
        logger.info("Video detection completed", 
                   file_name=file.filename,
                   frames_analyzed=len(result.frame_results) if result.frame_results else 0)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Video detection failed", error=str(e), file_name=file.filename)
        raise HTTPException(status_code=500, detail="Video detection failed")


@router.get("/history", response_model=DetectionHistoryList)
async def get_detection_history(
    skip: int = 0,
    limit: int = 100,
    prediction: Optional[str] = None,
    model_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get detection history
    """
    try:
        detection_service = DetectionService(db)
        history = await detection_service.get_history(
            skip=skip,
            limit=limit,
            prediction=prediction,
            model_type=model_type
        )
        return history
        
    except Exception as e:
        logger.error("Failed to get detection history", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get detection history")


@router.get("/statistics", response_model=DetectionStatistics)
async def get_detection_statistics(db: Session = Depends(get_db)):
    """
    Get detection statistics
    """
    try:
        detection_service = DetectionService(db)
        stats = await detection_service.get_statistics()
        return stats
        
    except Exception as e:
        logger.error("Failed to get detection statistics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get detection statistics")


@router.get("/models")
async def get_available_models():
    """
    Get available detection models
    """
    try:
        from app.models.ml_models import ModelRegistry
        models = ModelRegistry.list_models()
        return {"models": models, "default": "vgg"}
        
    except Exception as e:
        logger.error("Failed to get available models", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get available models")


@router.delete("/history/{detection_id}")
async def delete_detection_record(
    detection_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete detection record
    """
    try:
        detection_service = DetectionService(db)
        success = await detection_service.delete_detection_record(detection_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Detection record not found")
        
        return {"message": "Detection record deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete detection record", error=str(e), detection_id=detection_id)
        raise HTTPException(status_code=500, detail="Failed to delete detection record")
