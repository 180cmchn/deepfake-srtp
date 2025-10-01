"""
Model schemas for deepfake detection platform
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class ModelStatus(str, Enum):
    TRAINING = "training"
    READY = "ready"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


class ModelInfo(BaseModel):
    """Schema for model information"""
    name: str = Field(..., min_length=1, max_length=255)
    model_type: str = Field(..., description="Model type")
    version: str = Field(..., min_length=1, max_length=50)
    description: Optional[str] = Field(None, max_length=1000)
    input_size: int = Field(default=224, ge=32, le=1024)
    num_classes: int = Field(default=2, ge=2, le=1000)
    parameters: Optional[Dict[str, Any]] = None
    
    @validator('model_type')
    def validate_model_type(cls, v):
        supported_models = ["vgg", "lrcn", "swin", "vit", "resnet"]
        if v not in supported_models:
            raise ValueError(f'Model type must be one of: {supported_models}')
        return v


class ModelCreate(ModelInfo):
    """Schema for creating model"""
    file_path: str = Field(..., description="Path to model file")
    training_job_id: Optional[int] = None


class ModelUpdate(BaseModel):
    """Schema for updating model"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    status: Optional[ModelStatus] = None
    is_default: Optional[bool] = None
    deployment_info: Optional[Dict[str, Any]] = None


class ModelMetrics(BaseModel):
    """Schema for model performance metrics"""
    accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    precision: Optional[float] = Field(None, ge=0.0, le=1.0)
    recall: Optional[float] = Field(None, ge=0.0, le=1.0)
    f1_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    auc_roc: Optional[float] = Field(None, ge=0.0, le=1.0)
    confusion_matrix: Optional[Dict[str, Any]] = None
    classification_report: Optional[Dict[str, Any]] = None


class ModelResponse(ModelInfo):
    """Schema for model response"""
    id: int
    file_path: str
    status: ModelStatus
    metrics: Optional[ModelMetrics] = None
    is_default: bool
    deployment_info: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime]
    training_job_id: Optional[int] = None
    
    class Config:
        from_attributes = True


class ModelList(BaseModel):
    """Schema for model list response"""
    models: List[ModelResponse]
    total: int
    page: int
    size: int
    pages: int


class ModelDeployment(BaseModel):
    """Schema for model deployment"""
    model_id: int
    deployment_config: Dict[str, Any]
    endpoint_url: Optional[str] = None
    health_check_url: Optional[str] = None
    deployment_status: str
    deployed_at: Optional[datetime] = None


class ModelComparison(BaseModel):
    """Schema for model comparison results"""
    model_id_1: int
    model_id_2: int
    comparison_metrics: Dict[str, Any]
    winner: str
    improvement_percentage: Optional[float] = None


class ModelStatistics(BaseModel):
    """Schema for model statistics"""
    total_models: int
    models_by_type: Dict[str, int]
    models_by_status: Dict[str, int]
    average_accuracy: Optional[float] = None
    best_model: Optional[ModelResponse] = None
    recent_deployments: List[ModelResponse]
