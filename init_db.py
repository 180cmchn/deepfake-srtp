#!/usr/bin/env python3
"""
Database initialization script for Deepfake Detection Platform
Based on ai-manager-plateform approach
"""
import sys
import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Add the project root to the Python path
sys.path.append(os.path.dirname(__file__))

from app.core.database import create_tables, engine, test_connection_with_retry
from app.core.config import settings
from app.core.logging import logger

def init_database():
    """Initialize the database with all tables"""
    try:
        # Mask password in logs for security
        db_url = settings.DATABASE_URL
        if '@' in db_url:
            masked_url = db_url.split('@')[0].split(':')[:-1] + ['***@'] + db_url.split('@')[1:]
            masked_url = ':'.join(masked_url[:-1]) + '@' + masked_url[-1]
        else:
            masked_url = db_url
            
        logger.info("Initializing database", database_url=masked_url)
        
        # Test database connection with retry
        if not test_connection_with_retry():
            logger.error("Database connection failed")
            return False
        
        logger.info("Database connection successful")
        
        # Create all tables with retry
        if not create_tables_with_retry():
            logger.error("Failed to create database tables")
            return False
        
        logger.info("Database tables created successfully")
        
        return True
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        return False


def create_tables_with_retry(max_retries: int = 3, retry_delay: int = 2) -> bool:
    """
    Create all database tables with retry mechanism
    """
    from app.core.database import Base
    import time
    
    for attempt in range(max_retries):
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created successfully")
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error("Failed to create database tables after retries", 
                           error=str(e), attempts=max_retries)
                return False
            logger.warning(f"Table creation attempt {attempt + 1} failed, retrying in {retry_delay}s...", 
                         error=str(e))
            time.sleep(retry_delay)
    return False


if __name__ == "__main__":
    print("Deepfake Detection Platform - Database Initialization")
    
    # Mask password in output
    db_url = settings.DATABASE_URL
    if '@' in db_url:
        masked_url = db_url.split('@')[0].split(':')[:-1] + ['***@'] + db_url.split('@')[1:]
        masked_url = ':'.join(masked_url[:-1]) + '@' + masked_url[-1]
    else:
        masked_url = db_url
    
    print(f"Database URL: {masked_url}")
    
    success = init_database()
    if not success:
        print("❌ Database initialization failed!")
        sys.exit(1)
    
    print("✅ Database initialized successfully!")
