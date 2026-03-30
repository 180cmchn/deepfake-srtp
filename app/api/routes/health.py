"""
Health check and system status API routes
"""

import sys
from datetime import datetime
from typing import Any, Dict, Optional

import os
import psutil
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.database import get_database_health_snapshot, get_db
from app.core.logging import logger
from app.core.auth import get_current_user, require_admin
from app.core.config import settings

router = APIRouter()
EXPECTED_STATUS_TABLES = (
    "detection_results",
    "training_jobs",
    "model_registry",
    "dataset_info",
)


def _response(payload: Dict[str, Any], *, status_code: int = 200) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=payload)


def _build_database_status(
    *,
    healthy: bool,
    response_time_ms: float,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    database_status: Dict[str, Any] = {
        "status": "healthy" if healthy else "unhealthy",
        "healthy": healthy,
        "response_time_ms": response_time_ms,
    }
    if error:
        database_status["error"] = error
    return database_status


@router.get("/health")
async def health_check():
    """
    Basic health check endpoint
    """
    try:
        db_health = get_database_health_snapshot()

        # Check disk space
        disk_usage = psutil.disk_usage("/")
        disk_free_percent = (disk_usage.free / disk_usage.total) * 100

        # Check memory
        memory = psutil.virtual_memory()

        checks = {
            "database": _build_database_status(
                healthy=db_health["healthy"],
                response_time_ms=db_health["response_time_ms"],
                error=db_health.get("error"),
            ),
            "disk_space": {
                "status": "healthy" if disk_free_percent > 10 else "warning",
                "free_percent": round(disk_free_percent, 2),
                "free_gb": round(disk_usage.free / (1024**3), 2),
            },
            "memory": {
                "status": "healthy" if memory.available > 1024**3 else "warning",
                "available_gb": round(memory.available / (1024**3), 2),
                "usage_percent": memory.percent,
            },
        }

        overall_status = "unhealthy" if not db_health["healthy"] else "healthy"
        if overall_status == "healthy" and any(
            check.get("status") == "warning"
            for name, check in checks.items()
            if name != "database"
        ):
            overall_status = "degraded"

        health_status = {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.APP_VERSION,
            "checks": checks,
        }

        return _response(
            health_status,
            status_code=200 if overall_status != "unhealthy" else 503,
        )

    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return _response(
            {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": settings.APP_VERSION,
                "error": str(e),
            },
            status_code=503,
        )


@router.get("/status")
async def get_system_status(
    db: Session = Depends(get_db), current_user: str = Depends(get_current_user)
):
    """
    Get detailed system status
    """
    try:
        db_health = get_database_health_snapshot()

        # Get system information
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        database_status = _build_database_status(
            healthy=db_health["healthy"],
            response_time_ms=db_health["response_time_ms"],
            error=db_health.get("error"),
        )
        if not db_health["healthy"]:
            status_payload = {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory": {
                        "total_gb": round(memory.total / (1024**3), 2),
                        "available_gb": round(memory.available / (1024**3), 2),
                        "used_gb": round(memory.used / (1024**3), 2),
                        "percent": memory.percent,
                    },
                    "disk": {
                        "total_gb": round(disk.total / (1024**3), 2),
                        "free_gb": round(disk.free / (1024**3), 2),
                        "used_gb": round(disk.used / (1024**3), 2),
                        "percent": round((disk.used / disk.total) * 100, 2),
                    },
                },
                "database": database_status,
                "application": {
                    "environment": getattr(settings, "ENVIRONMENT", "development"),
                    "debug": getattr(settings, "DEBUG", False),
                    "log_level": getattr(settings, "LOG_LEVEL", "INFO"),
                    "version": settings.APP_VERSION,
                },
            }
            return _response(status_payload, status_code=503)

        # Get database statistics
        db_stats = {}
        database_overall_status = "healthy"
        for table in EXPECTED_STATUS_TABLES:
            try:
                result = db.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = result.scalar()
                db_stats[table] = {
                    "status": "healthy",
                    "row_count": int(count or 0),
                }
            except Exception as table_error:
                database_overall_status = "degraded"
                db_stats[table] = {
                    "status": "unhealthy",
                    "row_count": None,
                    "error": str(table_error),
                }
                logger.error(
                    "Failed to get health table count",
                    table=table,
                    error=str(table_error),
                )

        top_level_status = (
            "healthy" if database_overall_status == "healthy" else "degraded"
        )
        status = {
            "status": top_level_status,
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "percent": memory.percent,
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2),
                    "percent": round((disk.used / disk.total) * 100, 2),
                },
            },
            "database": {
                **database_status,
                "status": database_overall_status,
                "tables": db_stats,
            },
            "application": {
                "environment": getattr(settings, "ENVIRONMENT", "development"),
                "debug": getattr(settings, "DEBUG", False),
                "log_level": getattr(settings, "LOG_LEVEL", "INFO"),
                "version": settings.APP_VERSION,
            },
        }

        return _response(status, status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get system status", error=str(e), user=current_user)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to get system status",
                "error": str(e),
            },
        )


@router.get("/logs")
async def get_application_logs(
    tail_lines: int = Query(
        100, ge=0, le=10000, description="Number of lines to return from end of log"
    ),
    level: Optional[str] = Query(
        None, description="Filter by log level (DEBUG, INFO, WARNING, ERROR)"
    ),
    current_user: str = Depends(require_admin),
):
    """
    Get application logs (admin only)
    """
    try:
        # This is a simplified implementation
        # In production, you might want to read from actual log files
        log_file_path = "logs/app.log"

        if not os.path.exists(log_file_path):
            return {
                "logs": [],
                "message": "No log file found",
                "tail_lines": tail_lines,
            }

        # Read last N lines from log file
        with open(log_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Get last tail_lines
        recent_lines = lines[-tail_lines:] if tail_lines > 0 else lines

        # Filter by level if specified
        if level:
            recent_lines = [
                line for line in recent_lines if level.upper() in line.upper()
            ]

        return {
            "logs": [line.strip() for line in recent_lines],
            "total_lines": len(recent_lines),
            "tail_lines": tail_lines,
            "level_filter": level,
        }

    except Exception as e:
        logger.error("Failed to get application logs", error=str(e), user=current_user)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to get application logs",
                "error": str(e),
            },
        )


@router.get("/metrics")
async def get_system_metrics(
    db: Session = Depends(get_db), current_user: str = Depends(get_current_user)
):
    """
    Get system metrics for monitoring
    """
    try:
        # Get basic system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Get network stats
        network = psutil.net_io_counters()

        # Get process count
        process_count = len(psutil.pids())

        # Get boot time
        boot_time = psutil.boot_time()

        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": round((disk.used / disk.total) * 100, 2),
                "process_count": process_count,
                "uptime_seconds": int(datetime.utcnow().timestamp() - boot_time),
            },
            "network": {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv,
            },
            "application": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": os.name,
                "pid": os.getpid(),
            },
        }

        return metrics

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get system metrics", error=str(e), user=current_user)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to get system metrics",
                "error": str(e),
            },
        )
