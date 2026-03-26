"""
Model Manager
High-level model lifecycle management
"""

from typing import Dict, Any, Optional
import torch
from app.core.config import settings
from app.core.logging import logger
from app.models.load_model import load_all_models
from app.models.registry import model_registry


class ModelManager:
    """
    Manages model lifecycle
    - Loading
    - Caching
    - GPU memory management
    """
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or settings.DEVICE
        self.use_amp = settings.USE_AMP
        self.models: Dict[str, Any] = {}
        self.loaded = False
    
    def load_models(self):
        """Load all models"""
        if self.loaded:
            logger.warning("Models already loaded")
            return
        
        logger.info("Loading models...")
        
        # Load all models
        self.models = load_all_models(self.device)
        
        # Register in global registry
        for name, model in self.models.items():
            model_registry.register(name, model)
        
        self.loaded = True
        logger.info("All models loaded and registered")
    
    def get_model(self, name: str) -> Optional[Any]:
        """Get model by name"""
        return self.models.get(name)
    
    def unload_models(self):
        """Unload models and free memory"""
        self.models.clear()
        model_registry.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.loaded = False
        logger.info("Models unloaded")
    
    def warmup(self, batch_size: int = 1):
        """
        Warmup models with dummy inputs
        Useful for CUDA initialization
        """
        if not self.loaded:
            self.load_models()
        
        logger.info("Warming up models...")
        
        # TODO: Create dummy inputs and run forward pass
        # This helps with CUDA initialization and timing
        
        logger.info("Warmup complete")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get GPU memory usage
        
        Returns:
            Dictionary with memory info
        """
        if not torch.cuda.is_available():
            return {"available": 0, "allocated": 0, "reserved": 0}
        
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2
        }
    
    def reset_memory_stats(self):
        """Reset GPU memory statistics"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
