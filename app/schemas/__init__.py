"""
Schemas module for deepfake detection platform
"""

from .training import *
from .detection import *
from .models import *
from .datasets import *

__all__ = [
    # Training schemas
    "TrainingJob", "TrainingJobCreate", "TrainingJobUpdate", "TrainingJobResponse",
    "TrainingParameters", "TrainingResults",
    
    # Detection schemas
    "DetectionRequest", "DetectionResponse", "DetectionResult", "BatchDetectionRequest",
    
    # Model schemas
    "ModelInfo", "ModelCreate", "ModelUpdate", "ModelResponse", "ModelList",
    "ModelMetrics", "ModelDeployment",
    
    # Dataset schemas
    "DatasetInfo", "DatasetCreate", "DatasetUpdate", "DatasetResponse", "DatasetStats"
]
