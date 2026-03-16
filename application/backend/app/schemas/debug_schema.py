"""
Debug & Monitoring Schemas
"""

from pydantic import BaseModel
from typing import Dict, List, Optional


class ModuleTimings(BaseModel):
    """Timing information for each pipeline module"""
    module1_preprocessing: float
    module2_feature: float
    module3_sequence: float
    module4_language: float
    total: float


class DebugInfo(BaseModel):
    """Debug information for inference"""
    input_shape: List[int]
    device: str
    memory_used: float
    gpu_utilization: Optional[float] = None
    timings: ModuleTimings
