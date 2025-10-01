"""
Database models for deepfake detection platform
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
from enum import Enum


class JobStatus(str, Enum):
    """Training job status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelStatus(str, Enum):
    """Model status enumeration"""
    TRAINING = "training"
    READY = "ready"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


class DetectionStatus(str, Enum):
    """Detection status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingJob(Base):
    """Training job model"""
    __tablename__ = "training_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    model_type = Column(String(50), nullable=False)
    dataset_path = Column(String(500), nullable=False)
    status = Column(String(20), default=JobStatus.PENDING.value)
    progress = Column(Float, default=0.0)
    
    # Training parameters
    epochs = Column(Integer, default=50)
    learning_rate = Column(Float, default=0.001)
    batch_size = Column(Integer, default=32)
    
    # Results
    accuracy = Column(Float)
    loss = Column(Float)
    model_path = Column(String(500))
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    error_message = Column(Text)
    
    # Soft delete
    del_flag = Column(Integer, default=0)
    
    # Relationships
    model_registry = relationship("ModelRegistry", back_populates="training_job", uselist=False)


class ModelRegistry(Base):
    """Model registry for tracking trained models"""
    __tablename__ = "model_registry"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True)
    model_type = Column(String(50), nullable=False)
    version = Column(String(50), nullable=False)
    file_path = Column(String(500), nullable=False)
    
    # Model metadata
    input_size = Column(Integer, default=224)
    num_classes = Column(Integer, default=2)
    parameters = Column(JSON)
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    
    # Status and deployment
    status = Column(String(20), default=ModelStatus.TRAINING.value)
    is_default = Column(Boolean, default=False)
    deployment_info = Column(JSON)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Foreign keys
    training_job_id = Column(Integer, ForeignKey("training_jobs.id"))
    
    # Soft delete
    del_flag = Column(Integer, default=0)
    
    # Relationships
    training_job = relationship("TrainingJob", back_populates="model_registry")
    detection_results = relationship("DetectionResult", back_populates="model")


class DetectionResult(Base):
    """Detection results for tracking deepfake detections"""
    __tablename__ = "detection_results"
    
    id = Column(Integer, primary_key=True, index=True)
    file_path = Column(String(500), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(20), nullable=False)  # image, video
    
    # Detection results
    prediction = Column(String(20), nullable=False)  # real, fake
    confidence = Column(Float, nullable=False)
    processing_time = Column(Float)  # in seconds
    
    # Additional metadata
    frame_number = Column(Integer)  # for video files
    resolution = Column(String(20))  # widthxheight
    file_size = Column(Integer)  # in bytes
    
    # Status
    status = Column(String(20), default=DetectionStatus.COMPLETED.value)
    error_message = Column(Text)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Foreign keys
    model_id = Column(Integer, ForeignKey("model_registry.id"))
    
    # Soft delete
    del_flag = Column(Integer, default=0)
    
    # Relationships
    model = relationship("ModelRegistry", back_populates="detection_results")


class DatasetInfo(Base):
    """Dataset information for tracking training and test data"""
    __tablename__ = "dataset_info"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text)
    path = Column(String(500), nullable=False)
    
    # Dataset statistics
    total_samples = Column(Integer)
    real_samples = Column(Integer)
    fake_samples = Column(Integer)
    train_samples = Column(Integer)
    val_samples = Column(Integer)
    test_samples = Column(Integer)
    
    # Dataset configuration
    image_size = Column(Integer, default=224)
    frame_extraction_interval = Column(Integer, default=4)
    max_frames_per_video = Column(Integer, default=20)
    
    # Processing status
    is_processed = Column(Boolean, default=False)
    processing_status = Column(String(20), default="pending")
    error_message = Column(Text)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Soft delete
    del_flag = Column(Integer, default=0)


class SystemConfig(Base):
    """System configuration"""
    __tablename__ = "system_config"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), nullable=False, unique=True)
    value = Column(Text)
    description = Column(Text)
    config_type = Column(String(20), default="string")  # string, int, float, bool, json
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Soft delete
    del_flag = Column(Integer, default=0)


class AuditLog(Base):
    """Audit log for tracking system operations"""
    __tablename__ = "audit_log"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100))
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(String(100))
    details = Column(JSON)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
