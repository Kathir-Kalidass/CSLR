"""
GPU Monitoring
Tracks GPU utilization and memory
"""

import torch
from typing import Any, Dict
import psutil


class GPUMonitor:
    """
    Monitors GPU resource usage
    """
    
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
    
    def get_gpu_memory(self) -> Dict[str, float]:
        """
        Get current GPU memory usage
        
        Returns:
            Dictionary with memory stats in MB
        """
        if not self.cuda_available:
            return {
                "allocated": 0,
                "reserved": 0,
                "free": 0,
                "total": 0
            }
        
        # Get memory in bytes, convert to MB
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        free = total - allocated
        
        return {
            "allocated_mb": round(allocated, 2),
            "reserved_mb": round(reserved, 2),
            "free_mb": round(free, 2),
            "total_mb": round(total, 2),
            "utilization_percent": round((allocated / total) * 100, 2) if total > 0 else 0
        }
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get GPU device information
        
        Returns:
            GPU device info
        """
        if not self.cuda_available:
            return {"available": False}
        
        props = torch.cuda.get_device_properties(0)
        
        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_gb": round(props.total_memory / 1024**3, 2),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version()
        }
    
    def get_system_memory(self) -> Dict[str, float]:
        """
        Get system RAM usage
        
        Returns:
            System memory stats
        """
        mem = psutil.virtual_memory()
        
        return {
            "total_gb": round(mem.total / 1024**3, 2),
            "available_gb": round(mem.available / 1024**3, 2),
            "used_gb": round(mem.used / 1024**3, 2),
            "percent": mem.percent
        }
    
    def get_full_status(self) -> Dict[str, Any]:
        """
        Get complete system status
        
        Returns:
            Full monitoring data
        """
        return {
            "gpu": self.get_gpu_info(),
            "gpu_memory": self.get_gpu_memory(),
            "system_memory": self.get_system_memory(),
            "cpu_percent": psutil.cpu_percent(interval=0.1)
        }
    
    @staticmethod
    def clear_gpu_cache():
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# Global monitor instance
gpu_monitor = GPUMonitor()
