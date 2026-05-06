"""
Database configuration for deepfake detection platform
"""

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from contextlib import contextmanager
from typing import Any, Dict, Generator
import time
from .config import settings
from .logging import logger


def create_database_engine():
    """Create the SQLite engine used by the application."""
    return create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
        pool_pre_ping=True,
    )


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


def get_database_health_snapshot() -> Dict[str, Any]:
    started_at = time.perf_counter()
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        response_time_ms = round((time.perf_counter() - started_at) * 1000, 2)
        return {
            "healthy": True,
            "status": "healthy",
            "response_time_ms": response_time_ms,
        }
    except Exception as e:
        response_time_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.error("Database health snapshot failed", error=str(e))
        return {
            "healthy": False,
            "status": "unhealthy",
            "response_time_ms": response_time_ms,
            "error": str(e),
        }


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
                logger.error(
                    "Database connection test failed after retries",
                    error=str(e),
                    attempts=max_retries,
                )
                return False
            logger.warning(
                f"Database connection attempt {attempt + 1} failed, retrying in {retry_delay}s...",
                error=str(e),
            )
            time.sleep(retry_delay)
        except SQLAlchemyError as e:
            logger.error(
                "Database connection test failed with SQLAlchemy error", error=str(e)
            )
            return False
        except Exception as e:
            logger.error(
                "Database connection test failed with unexpected error", error=str(e)
            )
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
                logger.error(
                    "Failed to create database tables after retries",
                    error=str(e),
                    attempts=max_retries,
                )
                return False
            logger.warning(
                f"Table creation attempt {attempt + 1} failed, retrying in {retry_delay}s...",
                error=str(e),
            )
            time.sleep(retry_delay)
        except SQLAlchemyError as e:
            logger.error(
                "Failed to create database tables with SQLAlchemy error", error=str(e)
            )
            raise
        except Exception as e:
            logger.error(
                "Failed to create database tables with unexpected error", error=str(e)
            )
            raise
    return False


def ensure_runtime_schema() -> bool:
    """Apply lightweight additive schema fixes for existing deployments."""
    try:
        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names())
        if "detection_results" not in existing_tables:
            return True

        existing_columns = {
            column["name"] for column in inspector.get_columns("detection_results")
        }
        statements = []

        if "model_name" not in existing_columns:
            statements.append(
                "ALTER TABLE detection_results ADD COLUMN model_name VARCHAR(255)"
            )
        if "model_type" not in existing_columns:
            statements.append(
                "ALTER TABLE detection_results ADD COLUMN model_type VARCHAR(50)"
            )
        if "source_total_frames" not in existing_columns:
            statements.append(
                "ALTER TABLE detection_results ADD COLUMN source_total_frames INTEGER"
            )
        if "source_fps" not in existing_columns:
            statements.append(
                "ALTER TABLE detection_results ADD COLUMN source_fps FLOAT"
            )
        if "source_duration_seconds" not in existing_columns:
            statements.append(
                "ALTER TABLE detection_results ADD COLUMN source_duration_seconds FLOAT"
            )
        if "sampled_frame_count" not in existing_columns:
            statements.append(
                "ALTER TABLE detection_results ADD COLUMN sampled_frame_count INTEGER"
            )
        if "analyzed_frame_count" not in existing_columns:
            statements.append(
                "ALTER TABLE detection_results ADD COLUMN analyzed_frame_count INTEGER"
            )
        if "sampled_duration_seconds" not in existing_columns:
            statements.append(
                "ALTER TABLE detection_results ADD COLUMN sampled_duration_seconds FLOAT"
            )

        if not statements:
            return True

        with engine.begin() as connection:
            for statement in statements:
                connection.execute(text(statement))

        logger.info(
            "Runtime schema adjustments applied",
            table="detection_results",
            statements=statements,
        )
        return True
    except Exception as e:
        logger.error("Failed to apply runtime schema adjustments", error=str(e))
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
