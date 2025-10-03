"""
Database configuration for deepfake detection platform
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from contextlib import contextmanager
from typing import Generator
import time
import sqlite3
import pymysql
from .config import settings
from .logging import logger

def create_database_engine():
    """Create database engine with optimized configuration based on ai-manager-plateform"""
    if settings.DATABASE_URL.startswith("sqlite"):
        engine = create_engine(
            settings.DATABASE_URL,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False,
            pool_pre_ping=True
        )
    else:
        engine = create_engine(
            settings.DATABASE_URL,
            pool_pre_ping=True,
            pool_recycle=300,    # Shorter recycle time like ai-manager-plateform
            pool_size=10,        # Base connection pool size
            max_overflow=20,     # Additional connections when needed
            echo=False,          # Set to True for SQL debugging
        )
    return engine


# Create database engine
engine = create_database_engine()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for all database models"""
    pass


def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """
    Create all database tables
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error("Failed to create database tables", error=str(e))
        raise


def test_connection() -> bool:
    """
    Test database connection
    """
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("Database connection test successful")
        return True
    except Exception as e:
        logger.error("Database connection test failed", error=str(e))
        return False


def test_connection_with_retry(max_retries: int = 3, retry_delay: int = 1) -> bool:
    """
    Test database connection with retry mechanism for transient errors
    """
    for attempt in range(max_retries):
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
        except OperationalError as e:
            if attempt == max_retries - 1:
                logger.error("Database connection test failed after retries", 
                           error=str(e), attempts=max_retries)
                return False
            logger.warning(f"Database connection attempt {attempt + 1} failed, retrying in {retry_delay}s...", 
                         error=str(e))
            time.sleep(retry_delay)
        except SQLAlchemyError as e:
            logger.error("Database connection test failed with SQLAlchemy error", error=str(e))
            return False
        except Exception as e:
            logger.error("Database connection test failed with unexpected error", error=str(e))
            return False
    return False


def create_tables_with_retry(max_retries: int = 3, retry_delay: int = 2) -> bool:
    """
    Create all database tables with retry mechanism
    """
    for attempt in range(max_retries):
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created successfully")
            return True
        except OperationalError as e:
            if attempt == max_retries - 1:
                logger.error("Failed to create database tables after retries", 
                           error=str(e), attempts=max_retries)
                return False
            logger.warning(f"Table creation attempt {attempt + 1} failed, retrying in {retry_delay}s...", 
                         error=str(e))
            time.sleep(retry_delay)
        except SQLAlchemyError as e:
            logger.error("Failed to create database tables with SQLAlchemy error", error=str(e))
            raise
        except Exception as e:
            logger.error("Failed to create database tables with unexpected error", error=str(e))
            raise
    return False


@contextmanager
def get_db_session():
    """
    Safe database session context manager for use in services
    Automatically handles commit/rollback and session cleanup
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error("Database session error, rolled back", error=str(e))
        raise
    finally:
        session.close()


@contextmanager
def transaction_scope():
    """
    Transaction context manager for complex operations
    Ensures atomic transactions with proper rollback on errors
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
        logger.info("Transaction committed successfully")
    except SQLAlchemyError as e:
        session.rollback()
        logger.error("Transaction failed, rolled back", error=str(e))
        raise
    except Exception as e:
        session.rollback()
        logger.error("Unexpected error in transaction, rolled back", error=str(e))
        raise
    finally:
        session.close()


def get_db_session_manual() -> Session:
    """
    Get a database session (for use in services)
    Note: Manual session management - user must handle commit/rollback/close
    """
    return SessionLocal()
