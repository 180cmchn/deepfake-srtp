"""
Health check and system status API routes
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Dict, Any, Optional
import psutil
import os
from datetime import datetime

from app.core.database import get_db
from app.core.logging import logger
from app.core.auth import get_current_user, require_admin
from app.core.config import settings

router = APIRouter()


@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """
    Basic health check endpoint
    """
    try:
        # Check database connection
        db_status = "healthy"
        try:
            db.execute(text("SELECT 1"))
        except Exception as e:
            db_status = f"unhealthy: {str(e)}"
            logger.error("Database health check failed", error=str(e))
        
        # Check disk space
        disk_usage = psutil.disk_usage('/')
        disk_free_percent = (disk_usage.free / disk_usage.total) * 100
        
        # Check memory
        memory = psutil.virtual_memory()
        
        health_status = {
            "status": "healthy" if db_status == "healthy" else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "checks": {
                "database": {
                    "status": db_status,
                    "response_time_ms": "< 100"  # TODO: Implement actual timing
                },
                "disk_space": {
                    "status": "healthy" if disk_free_percent > 10 else "warning",
                    "free_percent": round(disk_free_percent, 2),
                    "free_gb": round(disk_usage.free / (1024**3), 2)
                },
                "memory": {
                    "status": "healthy" if memory.available > 1024**3 else "warning",  # 1GB
                    "available_gb": round(memory.available / (1024**3), 2),
                    "usage_percent": memory.percent
                }
            }
        }
        
        return health_status
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.get("/status")
async def get_system_status(
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Get detailed system status
    """
    try:
        # Get system information
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get database statistics
        db_stats = {}
        try:
            # Get table counts
            tables = ['detection_records', 'training_jobs', 'models', 'datasets']
            for table in tables:
                try:
                    result = db.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.scalar()
                    db_stats[table] = count
                except Exception:
                    db_stats[table] = 0
        except Exception as e:
            logger.error("Failed to get database stats", error=str(e))
        
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "percent": memory.percent
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2),
                    "percent": round((disk.used / disk.total) * 100, 2)
                }
            },
            "database": db_stats,
            "application": {
                "environment": getattr(settings, 'ENVIRONMENT', 'development'),
                "debug": getattr(settings, 'DEBUG', False),
                "log_level": getattr(settings, 'LOG_LEVEL', 'INFO')
            }
        }
        
        return status
        
    except Exception as e:
        logger.error("Failed to get system status", error=str(e), user=current_user)
        raise Exception(f"Failed to get system status: {str(e)}")


@router.get("/logs")
async def get_application_logs(
    tail_lines: int = Query(100, ge=0, le=10000, description="Number of lines to return from end of log"),
    level: Optional[str] = Query(None, description="Filter by log level (DEBUG, INFO, WARNING, ERROR)"),
    current_user: str = Depends(require_admin)
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
                "tail_lines": tail_lines
            }
        
        # Read last N lines from log file
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Get last tail_lines
        recent_lines = lines[-tail_lines:] if tail_lines > 0 else lines
        
        # Filter by level if specified
        if level:
            recent_lines = [line for line in recent_lines if level.upper() in line.upper()]
        
        return {
            "logs": [line.strip() for line in recent_lines],
            "total_lines": len(recent_lines),
            "tail_lines": tail_lines,
            "level_filter": level
        }
        
    except Exception as e:
        logger.error("Failed to get application logs", error=str(e), user=current_user)
        raise Exception(f"Failed to get application logs: {str(e)}")


@router.get("/metrics")
async def get_system_metrics(
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Get system metrics for monitoring
    """
    try:
        # Get basic system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
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
                "uptime_seconds": int(datetime.utcnow().timestamp() - boot_time)
            },
            "network": {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            },
            "application": {
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                "platform": os.name,
                "pid": os.getpid()
            }
        }
        
        return metrics
        
    except Exception as e:
        logger.error("Failed to get system metrics", error=str(e), user=current_user)
        raise Exception(f"Failed to get system metrics: {str(e)}")
