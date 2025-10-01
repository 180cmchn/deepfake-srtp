"""
Configuration management for deepfake detection platform
"""

from typing import List, Optional
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="allow"
    )
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Deepfake Detection Platform"
    
    # Security Settings
    SECRET_KEY: str = "your-secret-key-change-this-in-production-make-it-very-long-and-random"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"
    
    # CORS Settings
    _BACKEND_CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8000"
    
    @property
    def BACKEND_CORS_ORIGINS(self) -> List[str]:
        """Convert comma-separated string to list of strings"""
        if hasattr(self, '_cors_origins_cache'):
            return self._cors_origins_cache
        
        origins_str = os.getenv('BACKEND_CORS_ORIGINS', self._BACKEND_CORS_ORIGINS)
        if isinstance(origins_str, str):
            self._cors_origins_cache = [origin.strip() for origin in origins_str.split(",") if origin.strip()]
        else:
            self._cors_origins_cache = ["http://localhost:3000", "http://localhost:8000"]
        
        return self._cors_origins_cache
    
    # Database Settings
    DATABASE_URL: str = "sqlite:///./deepfake_detection.db"
    
    # Redis Settings (for Celery)
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Celery Settings
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # File Storage
    UPLOAD_DIR: str = "./uploads"
    MODEL_DIR: str = "./models"
    LOG_DIR: str = "./logs"
    DATA_DIR: str = "./data"
    
    # Model Settings
    DEFAULT_MODEL_TYPE: str = "vgg"
    SUPPORTED_MODELS: List[str] = ["vgg", "lrcn", "swin", "vit", "resnet"]
    MODEL_BATCH_SIZE: int = 32
    MODEL_INPUT_SIZE: int = 224
    
    # Training Settings
    MAX_CONCURRENT_TRAINING_JOBS: int = 2
    TRAINING_TIMEOUT: int = 7200  # 2 hours
    DEFAULT_EPOCHS: int = 50
    DEFAULT_LEARNING_RATE: float = 0.001
    
    # Detection Settings
    DETECTION_TIMEOUT: int = 30  # 30 seconds
    CONFIDENCE_THRESHOLD: float = 0.5
    
    # Data Processing Settings
    FRAME_EXTRACTION_INTERVAL: int = 4
    MAX_FRAMES_PER_VIDEO: int = 20
    FACE_DETECTION_CONFIDENCE: float = 0.9
    
    @field_validator('MAX_CONCURRENT_TRAINING_JOBS')
    @classmethod
    def validate_training_jobs(cls, v):
        if v < 1:
            raise ValueError('MAX_CONCURRENT_TRAINING_JOBS must be at least 1')
        return v
    
    @field_validator('TRAINING_TIMEOUT')
    @classmethod
    def validate_training_timeout(cls, v):
        if v < 60:  # At least 1 minute
            raise ValueError('TRAINING_TIMEOUT must be at least 60 seconds')
        return v
    
    @field_validator('CONFIDENCE_THRESHOLD')
    @classmethod
    def validate_confidence_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('CONFIDENCE_THRESHOLD must be between 0.0 and 1.0')
        return v
    
    @field_validator('DEFAULT_LEARNING_RATE')
    @classmethod
    def validate_learning_rate(cls, v):
        if v <= 0:
            raise ValueError('DEFAULT_LEARNING_RATE must be positive')
        return v
    
    @model_validator(mode='after')
    def validate_urls(self):
        """Validate URL formats"""
        # Validate database URL
        if not self.DATABASE_URL.startswith(('sqlite:///', 'mysql+pymysql://', 'postgresql://')):
            raise ValueError('DATABASE_URL must be a valid database connection string')
        
        # Validate Redis URLs
        redis_urls = [self.REDIS_URL, self.CELERY_BROKER_URL, self.CELERY_RESULT_BACKEND]
        for url in redis_urls:
            if not url.startswith('redis://'):
                raise ValueError(f'Redis URL must start with redis://: {url}')
        
        return self


# Initialize settings instance
settings = Settings()
