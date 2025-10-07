"""
Authentication and authorization utilities
"""

from fastapi import Header, HTTPException, Depends
from typing import Optional
import structlog

logger = structlog.get_logger()


def get_current_user(x_user_id: str = Header(None, alias="X-User-ID")) -> str:
    """
    Extract current user from header. In production, this would integrate with your auth system.
    For now, we'll use a simple header-based approach similar to the reference project.
    """
    if not x_user_id:
        # For development, allow default user
        x_user_id = "system"
        logger.warning("No X-User-ID header provided, using default user", user=x_user_id)
    
    return x_user_id


def get_optional_user(x_user_id: str = Header(None, alias="X-User-ID")) -> Optional[str]:
    """
    Extract optional user from header. Returns None if not provided.
    """
    return x_user_id


def require_admin(current_user: str = Depends(get_current_user)) -> str:
    """
    Require admin user. For now, we'll consider 'admin' and 'system' as admin users.
    """
    if current_user not in ['admin', 'system']:
        raise HTTPException(
            status_code=403, 
            detail="Admin access required"
        )
    return current_user
