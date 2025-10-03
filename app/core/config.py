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
    
    # Application Settings
    APP_NAME: str = "Deepfake Detection Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    RELOAD: bool = False
    
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
    
    # MySQL Settings (for MySQL database)
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_DATABASE: str = "deepfake_detection"
    
    @property
    def MYSQL_URL(self) -> str:
        """Generate MySQL connection URL"""
        return f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
    
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
    MODELS_DIR: str = "./models"  # Added for consistency with .env
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # Model Settings
    DEFAULT_MODEL_TYPE: str = "vgg"
    _SUPPORTED_MODELS: str = "vgg,lrcn,swin,vit,resnet"
    MODEL_BATCH_SIZE: int = 32
    MODEL_INPUT_SIZE: int = 224
    
    @property
    def SUPPORTED_MODELS(self) -> List[str]:
        """Convert comma-separated string to list of strings"""
        if hasattr(self, '_supported_models_cache'):
            return self._supported_models_cache
        
        models_str = os.getenv('SUPPORTED_MODELS', self._SUPPORTED_MODELS)
        if isinstance(models_str, str):
            self._supported_models_cache = [model.strip() for model in models_str.split(",") if model.strip()]
        else:
            self._supported_models_cache = ["vgg", "lrcn", "swin", "vit", "resnet"]
        
        return self._supported_models_cache
    
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
    
    @field_validator('PORT', 'WORKERS', mode='before')
    @classmethod
    def convert_str_to_int(cls, v):
        """Convert string values to integers for environment variables"""
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                raise ValueError(f'Unable to convert {v} to integer')
        return v
    
    @field_validator('RELOAD', 'DEBUG', mode='before')
    @classmethod
    def convert_str_to_bool(cls, v):
        """Convert string values to booleans for environment variables"""
        if isinstance(v, str):
            return v.lower() in ('true', '1', 'yes', 'on')
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
