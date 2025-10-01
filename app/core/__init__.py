"""
Core module for deepfake detection platform
"""

from .config import settings
from .database import get_db, create_tables, test_connection
from .logging import logger

__all__ = ["settings", "get_db", "create_tables", "test_connection", "logger"]
