"""
Common Response Schemas
"""

from pydantic import BaseModel
from typing import Optional, Any


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: Optional[str] = None
    type: Optional[str] = None


class SuccessResponse(BaseModel):
    """Standard success response"""
    success: bool
    message: str
    data: Optional[Any] = None
