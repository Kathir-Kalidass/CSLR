"""
Environment utilities
GPU detection, system info, etc.
"""

import torch
import platform
from typing import Dict, Any


def get_system_info() -> Dict[str, Any]:
    """Get system and environment information"""
    
    info = {
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        })
    
    return info


def check_gpu_availability() -> bool:
    """Check if GPU is available for inference"""
    return torch.cuda.is_available()


def get_optimal_device() -> str:
    """Get optimal device (cuda/cpu) based on availability"""
    return "cuda" if torch.cuda.is_available() else "cpu"
