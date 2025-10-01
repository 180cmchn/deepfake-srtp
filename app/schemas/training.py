"""
Training schemas for deepfake detection platform
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingParameters(BaseModel):
    """Training parameters schema"""
    epochs: int = Field(default=50, ge=1, le=1000, description="Number of training epochs")
    learning_rate: float = Field(default=0.001, ge=0.0001, le=1.0, description="Learning rate")
    batch_size: int = Field(default=32, ge=1, le=256, description="Batch size")
    validation_split: float = Field(default=0.2, ge=0.1, le=0.5, description="Validation split ratio")
    early_stopping: bool = Field(default=True, description="Enable early stopping")
    patience: int = Field(default=10, ge=1, le=100, description="Early stopping patience")
    
    @validator('validation_split')
    def validate_validation_split(cls, v):
        if not 0.1 <= v <= 0.5:
            raise ValueError('Validation split must be between 0.1 and 0.5')
        return v


class TrainingResults(BaseModel):
    """Training results schema"""
    accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Final accuracy")
    loss: Optional[float] = Field(None, ge=0.0, description="Final loss")
    val_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Final validation accuracy")
    val_loss: Optional[float] = Field(None, ge=0.0, description="Final validation loss")
    training_time: Optional[float] = Field(None, ge=0.0, description="Training time in seconds")
    epochs_trained: Optional[int] = Field(None, ge=0, description="Number of epochs actually trained")
    best_epoch: Optional[int] = Field(None, ge=0, description="Best epoch number")
    model_path: Optional[str] = Field(None, description="Path to saved model")


class TrainingJobBase(BaseModel):
    """Base training job schema"""
    name: str = Field(..., min_length=1, max_length=255, description="Job name")
    description: Optional[str] = Field(None, max_length=1000, description="Job description")
    model_type: str = Field(..., description="Model type (vgg, lrcn, swin, vit, resnet)")
    dataset_path: str = Field(..., description="Path to dataset")
    parameters: Optional[TrainingParameters] = Field(default_factory=TrainingParameters)
    
    @validator('model_type')
    def validate_model_type(cls, v):
        supported_models = ["vgg", "lrcn", "swin", "vit", "resnet"]
        if v not in supported_models:
            raise ValueError(f'Model type must be one of: {supported_models}')
        return v


class TrainingJobCreate(TrainingJobBase):
    """Schema for creating training job"""
    pass


class TrainingJobUpdate(BaseModel):
    """Schema for updating training job"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    status: Optional[JobStatus] = None
    progress: Optional[float] = Field(None, ge=0.0, le=100.0)
    results: Optional[TrainingResults] = None
    error_message: Optional[str] = Field(None, max_length=1000)


class TrainingJobResponse(TrainingJobBase):
    """Schema for training job response"""
    id: int
    status: JobStatus
    progress: float
    parameters: TrainingParameters
    results: Optional[TrainingResults]
    created_at: datetime
    updated_at: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]
    
    class Config:
        from_attributes = True


class TrainingJobList(BaseModel):
    """Schema for training job list response"""
    jobs: list[TrainingJobResponse]
    total: int
    page: int
    size: int
    pages: int


class TrainingProgress(BaseModel):
    """Schema for training progress updates"""
    job_id: int
    status: JobStatus
    progress: float = Field(..., ge=0.0, le=100.0)
    current_epoch: Optional[int] = Field(None, ge=0)
    total_epochs: Optional[int] = Field(None, ge=0)
    current_loss: Optional[float] = Field(None, ge=0.0)
    current_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    estimated_time_remaining: Optional[int] = Field(None, ge=0, description="Estimated time remaining in seconds")
    message: Optional[str] = None


class TrainingLog(BaseModel):
    """Schema for training log entries"""
    job_id: int
    timestamp: datetime
    level: str = Field(..., description="Log level (info, warning, error)")
    message: str
    epoch: Optional[int] = None
    metrics: Optional[Dict[str, Any]] = None


class TrainingConfig(BaseModel):
    """Schema for global training configuration"""
    max_concurrent_jobs: int = Field(default=2, ge=1, le=10, description="Maximum concurrent training jobs")
    default_epochs: int = Field(default=50, ge=1, le=1000, description="Default number of epochs")
    default_learning_rate: float = Field(default=0.001, ge=0.0001, le=1.0, description="Default learning rate")
    default_batch_size: int = Field(default=32, ge=1, le=256, description="Default batch size")
    gpu_enabled: bool = Field(default=True, description="Enable GPU training")
    memory_limit: Optional[int] = Field(None, ge=1024, description="Memory limit in MB")
    timeout: int = Field(default=7200, ge=300, description="Training timeout in seconds")
    
    @validator('memory_limit')
    def validate_memory_limit(cls, v):
        if v is not None and v < 1024:
            raise ValueError('Memory limit must be at least 1024 MB')
        return v


class TrainingMetrics(BaseModel):
    """Schema for training metrics aggregation"""
    total_jobs: int
    active_jobs: int
    completed_jobs: int
    failed_jobs: int
    average_accuracy: Optional[float] = None
    average_training_time: Optional[float] = None
    success_rate: float = Field(..., ge=0.0, le=1.0)
    jobs_by_model_type: Dict[str, int]
    jobs_by_status: Dict[str, int]
