"""
Dataset management API routes for deepfake detection platform
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import uuid
from pathlib import Path

from app.core.database import get_db
from app.core.logging import logger
from app.schemas.datasets import (
    DatasetCreate, DatasetResponse, DatasetList, DatasetUpdate,
    DatasetProcessingConfig, DatasetFileAddRequest, DatasetFileAddResponse
)
from app.services.dataset_service import DatasetService

router = APIRouter()

# Supported file types
ALLOWED_IMAGE_TYPES = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
ALLOWED_VIDEO_TYPES = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
ALLOWED_TYPES = ALLOWED_IMAGE_TYPES | ALLOWED_VIDEO_TYPES

# File size limits (in bytes)
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
MAX_IMAGE_SIZE = 50 * 1024 * 1024   # 50MB for images
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500MB for videos


def validate_file(file: UploadFile) -> tuple[str, str]:
    """Validate file type and size"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_extension}. "
                   f"Allowed types: {', '.join(sorted(ALLOWED_TYPES))}"
        )
    
    # Determine file type
    file_type = "image" if file_extension in ALLOWED_IMAGE_TYPES else "video"
    
    # Check file size
    if hasattr(file, 'size') and file.size:
        if file_type == "image" and file.size > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Image file too large. Maximum size: {MAX_IMAGE_SIZE // (1024*1024)}MB"
            )
        elif file_type == "video" and file.size > MAX_VIDEO_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Video file too large. Maximum size: {MAX_VIDEO_SIZE // (1024*1024)}MB"
            )
        elif file.size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
    
    return file_type, file_extension


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
    file_path = None
    try:
        # Validate file type and size
        file_type, file_extension = validate_file(file)
        
        # Generate unique filename with type prefix
        type_prefix = "img" if file_type == "image" else "vid"
        unique_filename = f"{type_prefix}_{uuid.uuid4()}{file_extension}"
        file_path = os.path.join("data", unique_filename)
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        # Read file content with size check
        content = await file.read()
        
        # Additional size check for files without size attribute
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        logger.info("File uploaded successfully", 
                   filename=file.filename, 
                   file_type=file_type, 
                   size=len(content),
                   saved_path=file_path)
        
        # Create dataset entry
        dataset_data = DatasetCreate(
            name=name or file.filename,
            description=description or f"Uploaded {file_type} file",
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
        
        logger.info("Dataset created and processing started", 
                   dataset_id=result.id, 
                   dataset_name=result.name)
        
        return result
        
    except HTTPException:
        # Clean up uploaded file if it exists
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info("Cleaned up uploaded file", file_path=file_path)
            except Exception as cleanup_error:
                logger.error("Failed to clean up file", 
                           file_path=file_path, 
                           error=str(cleanup_error))
        raise
    except Exception as e:
        logger.error("Failed to upload dataset", 
                   error=str(e), 
                   filename=file.filename if file else "Unknown")
        
        # Clean up uploaded file if it exists
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info("Cleaned up uploaded file after error", file_path=file_path)
            except Exception as cleanup_error:
                logger.error("Failed to clean up file after error", 
                           file_path=file_path, 
                           error=str(cleanup_error))
        
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to upload dataset: {str(e)}"
        )


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


@router.post("/{dataset_id}/add-files", response_model=DatasetFileAddResponse)
async def add_files_to_dataset(
    dataset_id: int,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    reprocess: bool = True,
    description: str = None,
    config: DatasetProcessingConfig = Depends(),
    db: Session = Depends(get_db)
):
    """Add files to existing dataset"""
    uploaded_files = []
    file_paths = []
    
    try:
        # Validate dataset exists
        dataset_service = DatasetService(db)
        dataset = await dataset_service.get_dataset(dataset_id)
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Process each file
        for file in files:
            # Validate file
            file_type, file_extension = validate_file(file)
            
            # Generate unique filename with dataset prefix
            type_prefix = "img" if file_type == "image" else "vid"
            unique_filename = f"dataset_{dataset_id}_{type_prefix}_{uuid.uuid4()}{file_extension}"
            file_path = os.path.join("data", unique_filename)
            
            # Ensure data directory exists
            os.makedirs("data", exist_ok=True)
            
            # Read file content
            content = await file.read()
            
            # Save uploaded file
            with open(file_path, "wb") as buffer:
                buffer.write(content)
            
            uploaded_files.append((file, file_path, file_type, len(content)))
            file_paths.append(file_path)
            
            logger.info("File uploaded for dataset", 
                       dataset_id=dataset_id,
                       filename=file.filename, 
                       file_type=file_type, 
                       size=len(content),
                       saved_path=file_path)
        
        # Create request object
        request = DatasetFileAddRequest(
            reprocess=reprocess,
            description=description
        )
        
        # Add files to dataset
        result = await dataset_service.add_files_to_dataset(dataset_id, uploaded_files, request)
        
        # Start processing in background if requested
        if reprocess:
            background_tasks.add_task(
                dataset_service.process_dataset,
                dataset_id,
                config.dict()
            )
            
            logger.info("Dataset processing started after adding files", 
                       dataset_id=dataset_id)
        
        return result
        
    except HTTPException:
        # Clean up uploaded files on error
        for file_path in file_paths:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info("Cleaned up uploaded file", file_path=file_path)
                except Exception as cleanup_error:
                    logger.error("Failed to clean up file", 
                               file_path=file_path, 
                               error=str(cleanup_error))
        raise
    except Exception as e:
        logger.error("Failed to add files to dataset", 
                   error=str(e), 
                   dataset_id=dataset_id)
        
        # Clean up uploaded files on error
        for file_path in file_paths:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info("Cleaned up uploaded file after error", file_path=file_path)
                except Exception as cleanup_error:
                    logger.error("Failed to clean up file after error", 
                               file_path=file_path, 
                               error=str(cleanup_error))
        
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to add files to dataset: {str(e)}"
        )


@router.get("/{dataset_id}/files")
async def get_dataset_files(dataset_id: int, db: Session = Depends(get_db)):
    """Get all files in a dataset"""
    try:
        dataset_service = DatasetService(db)
        
        # Check if dataset exists
        dataset = await dataset_service.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Get files
        files = await dataset_service.get_dataset_files(dataset_id)
        
        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset.name,
            "files": files,
            "total_files": len(files)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get dataset files", error=str(e), dataset_id=dataset_id)
        raise HTTPException(status_code=500, detail="Failed to get dataset files")
