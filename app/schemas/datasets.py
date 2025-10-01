"""
Dataset schemas for deepfake detection platform
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime


class DatasetInfo(BaseModel):
    """Schema for dataset information"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    path: str = Field(..., description="Path to dataset")
    image_size: int = Field(default=224, ge=32, le=1024)
    frame_extraction_interval: int = Field(default=4, ge=1, le=30)
    max_frames_per_video: int = Field(default=20, ge=1, le=100)


class DatasetCreate(DatasetInfo):
    """Schema for creating dataset"""
    pass


class DatasetUpdate(BaseModel):
    """Schema for updating dataset"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    image_size: Optional[int] = Field(None, ge=32, le=1024)
    frame_extraction_interval: Optional[int] = Field(None, ge=1, le=30)
    max_frames_per_video: Optional[int] = Field(None, ge=1, le=100)


class DatasetStats(BaseModel):
    """Schema for dataset statistics"""
    total_samples: Optional[int] = None
    real_samples: Optional[int] = None
    fake_samples: Optional[int] = None
    train_samples: Optional[int] = None
    val_samples: Optional[int] = None
    test_samples: Optional[int] = None
    class_distribution: Optional[Dict[str, int]] = None
    data_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class DatasetResponse(DatasetInfo):
    """Schema for dataset response"""
    id: int
    stats: Optional[DatasetStats] = None
    is_processed: bool
    processing_status: str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class DatasetList(BaseModel):
    """Schema for dataset list response"""
    datasets: List[DatasetResponse]
    total: int
    page: int
    size: int
    pages: int


class DatasetProcessingConfig(BaseModel):
    """Schema for dataset processing configuration"""
    face_detection_confidence: float = Field(default=0.9, ge=0.5, le=1.0)
    min_face_size: int = Field(default=50, ge=10, le=500)
    max_face_size: int = Field(default=500, ge=50, le=2000)
    augment_data: bool = Field(default=True)
    augmentation_factor: int = Field(default=2, ge=1, le=10)
    balance_classes: bool = Field(default=True)
    validation_split: float = Field(default=0.2, ge=0.1, le=0.5)
    test_split: float = Field(default=0.1, ge=0.05, le=0.3)
