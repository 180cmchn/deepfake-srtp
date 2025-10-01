"""
Detection schemas for deepfake detection platform
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class DetectionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PredictionType(str, Enum):
    REAL = "real"
    FAKE = "fake"


class FileType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class DetectionRequest(BaseModel):
    """Schema for detection request"""
    model_id: Optional[int] = Field(None, description="Model ID to use for detection")
    model_type: Optional[str] = Field(None, description="Model type (if model_id not provided)")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence threshold")
    return_probabilities: bool = Field(default=False, description="Return class probabilities")
    preprocess: bool = Field(default=True, description="Apply preprocessing")
    
    @validator('model_type')
    def validate_model_type(cls, v):
        if v is not None:
            supported_models = ["vgg", "lrcn", "swin", "vit", "resnet"]
            if v not in supported_models:
                raise ValueError(f'Model type must be one of: {supported_models}')
        return v


class DetectionResult(BaseModel):
    """Schema for single detection result"""
    prediction: PredictionType
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities")
    processing_time: float = Field(..., ge=0.0, description="Processing time in seconds")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model information used")
    
    @validator('probabilities')
    def validate_probabilities(cls, v):
        if v is not None:
            total = sum(v.values())
            if not abs(total - 1.0) < 0.01:  # Allow small floating point errors
                raise ValueError('Probabilities must sum to 1.0')
        return v


class DetectionResponse(BaseModel):
    """Schema for detection response"""
    success: bool
    file_info: Dict[str, Any]
    result: Optional[DetectionResult] = None
    error_message: Optional[str] = None
    processing_time: float = Field(..., ge=0.0)
    created_at: datetime
    
    class Config:
        from_attributes = True


class BatchDetectionRequest(BaseModel):
    """Schema for batch detection request"""
    file_paths: List[str] = Field(..., min_items=1, max_items=100, description="List of file paths")
    model_id: Optional[int] = Field(None, description="Model ID to use for detection")
    model_type: Optional[str] = Field(None, description="Model type (if model_id not provided)")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence threshold")
    return_probabilities: bool = Field(default=False, description="Return class probabilities")
    preprocess: bool = Field(default=True, description="Apply preprocessing")
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    max_workers: int = Field(default=4, ge=1, le=10, description="Maximum number of workers")
    
    @validator('model_type')
    def validate_model_type(cls, v):
        if v is not None:
            supported_models = ["vgg", "lrcn", "swin", "vit", "resnet"]
            if v not in supported_models:
                raise ValueError(f'Model type must be one of: {supported_models}')
        return v


class BatchDetectionResponse(BaseModel):
    """Schema for batch detection response"""
    success: bool
    total_files: int
    processed_files: int
    failed_files: int
    results: List[DetectionResponse]
    summary: Dict[str, Any]
    processing_time: float = Field(..., ge=0.0)
    created_at: datetime


class VideoDetectionRequest(BaseModel):
    """Schema for video detection request"""
    video_path: str = Field(..., description="Path to video file")
    model_id: Optional[int] = Field(None, description="Model ID to use for detection")
    model_type: Optional[str] = Field(None, description="Model type (if model_id not provided)")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence threshold")
    frame_extraction_interval: int = Field(default=4, ge=1, le=30, description="Frame extraction interval")
    max_frames: int = Field(default=20, ge=1, le=100, description="Maximum frames to analyze")
    aggregate_results: bool = Field(default=True, description="Aggregate frame results")
    return_frame_results: bool = Field(default=False, description="Return individual frame results")
    preprocess: bool = Field(default=True, description="Apply preprocessing")
    
    @validator('model_type')
    def validate_model_type(cls, v):
        if v is not None:
            supported_models = ["vgg", "lrcn", "swin", "vit", "resnet"]
            if v not in supported_models:
                raise ValueError(f'Model type must be one of: {supported_models}')
        return v


class FrameDetectionResult(BaseModel):
    """Schema for frame detection result"""
    frame_number: int
    timestamp: Optional[float] = Field(None, description="Timestamp in seconds")
    result: DetectionResult


class VideoDetectionResponse(BaseModel):
    """Schema for video detection response"""
    success: bool
    video_info: Dict[str, Any]
    aggregated_result: Optional[DetectionResult] = None
    frame_results: Optional[List[FrameDetectionResult]] = None
    summary: Dict[str, Any]
    processing_time: float = Field(..., ge=0.0)
    created_at: datetime


class DetectionHistory(BaseModel):
    """Schema for detection history"""
    id: int
    file_name: str
    file_type: FileType
    prediction: PredictionType
    confidence: float
    processing_time: float
    model_name: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class DetectionHistoryList(BaseModel):
    """Schema for detection history list"""
    detections: List[DetectionHistory]
    total: int
    page: int
    size: int
    pages: int


class DetectionStatistics(BaseModel):
    """Schema for detection statistics"""
    total_detections: int
    real_detections: int
    fake_detections: int
    average_confidence: float
    average_processing_time: float
    detections_by_model: Dict[str, int]
    detections_by_file_type: Dict[str, int]
    confidence_distribution: Dict[str, int]
    daily_detections: Dict[str, int]


class DetectionConfig(BaseModel):
    """Schema for detection configuration"""
    default_model_type: str = Field(default="vgg", description="Default model type")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Default confidence threshold")
    max_file_size: int = Field(default=100, ge=1, le=1000, description="Maximum file size in MB")
    supported_image_formats: List[str] = Field(default=["jpg", "jpeg", "png", "bmp", "tiff"])
    supported_video_formats: List[str] = Field(default=["mp4", "avi", "mov", "mkv", "wmv"])
    max_batch_size: int = Field(default=100, ge=1, le=1000, description="Maximum batch size")
    timeout: int = Field(default=30, ge=5, le=300, description="Detection timeout in seconds")
    enable_gpu: bool = Field(default=True, description="Enable GPU acceleration")
    
    @validator('default_model_type')
    def validate_default_model_type(cls, v):
        supported_models = ["vgg", "lrcn", "swin", "vit", "resnet"]
        if v not in supported_models:
            raise ValueError(f'Default model type must be one of: {supported_models}')
        return v


class DetectionError(BaseModel):
    """Schema for detection error"""
    error_code: str
    error_message: str
    file_path: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime


class ModelComparison(BaseModel):
    """Schema for model comparison results"""
    model_results: Dict[str, DetectionResult]
    consensus_prediction: PredictionType
    average_confidence: float
    confidence_variance: float
    agreement_score: float = Field(..., ge=0.0, le=1.0, description="Model agreement score")
    recommendation: str
