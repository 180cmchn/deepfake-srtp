#!/usr/bin/env python3
"""Database initialization script for Deepfake Detection Platform.

Default behavior is Alembic-first to avoid schema drift from raw create_all.
"""

import argparse
import os
import sys

from alembic import command
from alembic.config import Config
from dotenv import load_dotenv
from sqlalchemy.engine import make_url

# Load environment variables first
load_dotenv()

# Add project root for script execution
sys.path.append(os.path.dirname(__file__))

from app.core.config import settings
from app.core.database import Base, engine, test_connection_with_retry
from app.core.logging import logger


def mask_database_url(raw_url: str) -> str:
    """Return a safe database URL for logs by hiding password."""
    try:
        parsed = make_url(raw_url)
        return parsed.render_as_string(hide_password=True)
    except Exception:
        return raw_url


def run_alembic_upgrade() -> bool:
    """Apply Alembic migrations to head."""
    try:
        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")
        logger.info("Alembic migration completed", revision="head")
        return True
    except Exception as exc:
        logger.error("Alembic migration failed", error=str(exc))
        return False


def fallback_create_all() -> bool:
    """Fallback path for emergency bootstrapping without Alembic.

    This is disabled by default and should only be used intentionally.
    """
    try:
        # Ensure models are imported so metadata is fully registered.
        import app.models.database_models  # noqa: F401

        Base.metadata.create_all(bind=engine)
        logger.warning("Fallback create_all executed; consider using Alembic migrations")
        return True
    except Exception as exc:
        logger.error("Fallback create_all failed", error=str(exc))
        return False


def init_database(use_fallback: bool = False) -> bool:
    """Initialize database by checking connectivity and applying migrations."""
    masked_url = mask_database_url(settings.DATABASE_URL)
    logger.info("Initializing database", database_url=masked_url)

    if not test_connection_with_retry():
        logger.error("Database connection failed")
        return False

    logger.info("Database connection successful")

    if run_alembic_upgrade():
        return True

    if use_fallback:
        logger.warning("Trying fallback create_all because --fallback-create-all is enabled")
        return fallback_create_all()

    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize database schema")
    parser.add_argument(
        "--fallback-create-all",
        action="store_true",
        help="Fallback to SQLAlchemy create_all if Alembic upgrade fails",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Deepfake Detection Platform - Database Initialization")
    print(f"Database URL: {mask_database_url(settings.DATABASE_URL)}")

    success = init_database(use_fallback=args.fallback_create_all)
    if not success:
        print("❌ Database initialization failed!")
        sys.exit(1)

    print("✅ Database initialized successfully!")
