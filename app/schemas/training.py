"""
Training schemas for deepfake detection platform
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel as PydanticBaseModel, Field, validator, ConfigDict
from datetime import datetime
from enum import Enum


class BaseModel(PydanticBaseModel):
    model_config = ConfigDict(protected_namespaces=())


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingParameters(BaseModel):
    """Training parameters schema"""

    epochs: int = Field(
        default=50, ge=1, le=1000, description="Number of training epochs"
    )
    learning_rate: float = Field(
        default=0.001, ge=0.0001, le=1.0, description="Learning rate"
    )
    batch_size: int = Field(default=32, ge=1, le=256, description="Batch size")
    validation_split: float = Field(
        default=0.2, ge=0.1, le=0.5, description="Validation split ratio"
    )
    early_stopping: bool = Field(default=True, description="Enable early stopping")
    patience: int = Field(
        default=10, ge=1, le=100, description="Early stopping patience"
    )
    early_stopping_min_delta: float = Field(
        default=0.002,
        ge=0.0,
        le=1.0,
        description="Minimum validation accuracy improvement to reset early stopping",
    )
    weight_decay: float = Field(
        default=0.0001, ge=0.0, le=1.0, description="AdamW weight decay"
    )
    label_smoothing: float = Field(
        default=0.05,
        ge=0.0,
        le=0.2,
        description="Cross-entropy label smoothing factor",
    )
    class_weight_strategy: str = Field(
        default="balanced",
        description="Class weighting strategy: none, balanced, fake_prior",
    )
    use_official_test_as_validation: bool = Field(
        default=False,
        description="Whether to use Celeb-DF official test list as the validation split",
    )
    training_device: str = Field(
        default="auto", description="Training device: auto, mps, cuda, cpu"
    )

    @validator("validation_split")
    def validate_validation_split(cls, v):
        if not 0.1 <= v <= 0.5:
            raise ValueError("Validation split must be between 0.1 and 0.5")
        return v

    @validator("training_device")
    def validate_training_device(cls, v):
        allowed = {"auto", "mps", "cuda", "cpu"}
        if v not in allowed:
            raise ValueError(f"training_device must be one of: {sorted(allowed)}")
        return v

    @validator("class_weight_strategy")
    def validate_class_weight_strategy(cls, v):
        allowed = {"none", "balanced", "fake_prior"}
        if v not in allowed:
            raise ValueError(f"class_weight_strategy must be one of: {sorted(allowed)}")
        return v


class ValidationSubgroupMetric(BaseModel):
    sample_count: int = Field(default=0, ge=0)
    video_count: int = Field(default=0, ge=0)
    sample_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    sample_loss: Optional[float] = Field(None, ge=0.0)
    video_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    video_loss: Optional[float] = Field(None, ge=0.0)


class TrainingResults(BaseModel):
    """Training results schema"""

    accuracy: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Deprecated legacy alias of the persisted video-level validation accuracy summary.",
    )
    loss: Optional[float] = Field(
        None,
        ge=0.0,
        description="Deprecated legacy alias of the persisted video-level validation loss summary.",
    )
    val_accuracy: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Video-level validation accuracy summary returned for the completed training job.",
    )
    val_loss: Optional[float] = Field(
        None,
        ge=0.0,
        description="Video-level validation loss summary returned for the completed training job.",
    )
    val_sample_accuracy: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Validation accuracy at sample or clip level"
    )
    val_sample_loss: Optional[float] = Field(
        None, ge=0.0, description="Validation loss at sample or clip level"
    )
    val_video_count: Optional[int] = Field(
        None, ge=0, description="Number of validation videos aggregated"
    )
    val_subgroup_metrics: Optional[Dict[str, ValidationSubgroupMetric]] = Field(
        None, description="Per-subgroup validation metrics"
    )
    val_subgroup_macro_accuracy: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Macro video accuracy across validation subgroups",
    )
    checkpoint_selection_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Score used to select the best checkpoint before val_accuracy/val_loss tie-breaks.",
    )
    training_time: Optional[float] = Field(
        None, ge=0.0, description="Training time in seconds"
    )
    epochs_trained: Optional[int] = Field(
        None, ge=0, description="Number of epochs actually trained"
    )
    best_epoch: Optional[int] = Field(
        None,
        ge=0,
        description="Epoch selected by checkpoint_selection_score, then validation accuracy/loss tie-breaks.",
    )
    model_path: Optional[str] = Field(None, description="Path to saved model")


class TrainingJobBase(BaseModel):
    """Base training job schema"""

    name: str = Field(..., min_length=1, max_length=255, description="Job name")
    description: Optional[str] = Field(
        None, max_length=1000, description="Job description"
    )
    model_type: str = Field(
        ..., description="Model type (vgg, lrcn, swin, vit, resnet)"
    )
    dataset_path: str = Field(..., description="Path to dataset")
    parameters: TrainingParameters = Field(default_factory=TrainingParameters)

    @validator("model_type")
    def validate_model_type(cls, v):
        supported_models = ["vgg", "lrcn", "swin", "vit", "resnet"]
        if v not in supported_models:
            raise ValueError(f"Model type must be one of: {supported_models}")
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
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    progress_message: Optional[str] = None
    preprocessing_stage: Optional[str] = None
    preprocessing_progress: Optional[float] = Field(None, ge=0.0, le=100.0)
    preprocessing_current: Optional[int] = Field(None, ge=0)
    preprocessing_total: Optional[int] = Field(None, ge=0)
    preprocessing_unit: Optional[str] = None
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
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_epoch: Optional[int] = Field(None, ge=0)
    total_epochs: Optional[int] = Field(None, ge=0)
    current_loss: Optional[float] = Field(None, ge=0.0)
    current_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    estimated_time_remaining: Optional[int] = Field(
        None, ge=0, description="Estimated time remaining in seconds"
    )
    message: Optional[str] = None
    preprocessing_stage: Optional[str] = None
    preprocessing_progress: Optional[float] = Field(None, ge=0.0, le=100.0)
    preprocessing_current: Optional[int] = Field(None, ge=0)
    preprocessing_total: Optional[int] = Field(None, ge=0)
    preprocessing_unit: Optional[str] = None


class TrainingLog(BaseModel):
    """Schema for training log entries"""

    job_id: int
    timestamp: datetime
    level: str = Field(..., description="Log level (info, warning, error)")
    message: str
    epoch: Optional[int] = None
    metrics: Optional[Dict[str, Any]] = None


class EpochMetricPoint(BaseModel):
    """Per-epoch training and validation metrics."""

    epoch: int = Field(..., ge=1)
    train_loss: Optional[float] = Field(None, ge=0.0)
    train_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    val_loss: Optional[float] = Field(None, ge=0.0)
    val_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    val_sample_loss: Optional[float] = Field(None, ge=0.0)
    val_sample_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    val_video_count: Optional[int] = Field(None, ge=0)
    val_subgroup_metrics: Optional[Dict[str, ValidationSubgroupMetric]] = Field(None)
    val_subgroup_macro_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    checkpoint_selection_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    learning_rate: Optional[float] = Field(None, ge=0.0)
    recorded_at: Optional[datetime] = None


class TrainingEpochMetricsResponse(BaseModel):
    """Epoch metric history for charting training progress."""

    job_id: int
    job_name: Optional[str] = None
    model_type: Optional[str] = None
    total_epochs: Optional[int] = Field(None, ge=0)
    completed_epochs: int = Field(default=0, ge=0)
    available: bool = False
    metrics: list[EpochMetricPoint] = Field(default_factory=list)


class TrainingConfig(BaseModel):
    """Schema for global training configuration"""

    max_concurrent_jobs: int = Field(
        default=2, ge=1, le=10, description="Maximum concurrent training jobs"
    )
    default_epochs: int = Field(
        default=50, ge=1, le=1000, description="Default number of epochs"
    )
    default_learning_rate: float = Field(
        default=0.001, ge=0.0001, le=1.0, description="Default learning rate"
    )
    default_batch_size: int = Field(
        default=32, ge=1, le=256, description="Default batch size"
    )
    gpu_enabled: bool = Field(default=True, description="Enable GPU training")
    memory_limit: Optional[int] = Field(None, ge=1024, description="Memory limit in MB")
    timeout: int = Field(
        default=7200, ge=300, description="Training timeout in seconds"
    )

    @validator("memory_limit")
    def validate_memory_limit(cls, v):
        if v is not None and v < 1024:
            raise ValueError("Memory limit must be at least 1024 MB")
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
