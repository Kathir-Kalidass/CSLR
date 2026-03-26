"""
Health Check Schemas
"""

from pydantic import BaseModel, Field
from typing import Optional


class HealthResponse(BaseModel):
    """Basic health check response"""
    status: str = Field(..., description="Health status")
    message: str = Field(..., description="Status message")


class SystemInfoResponse(BaseModel):
    """Detailed system information"""
    platform: str
    python_version: str
    pytorch_version: str
    cuda_available: bool
    cuda_version: Optional[str] = None
    cudnn_version: Optional[int] = None
    gpu_count: Optional[int] = None
    gpu_name: Optional[str] = None
    gpu_memory_total: Optional[str] = None
