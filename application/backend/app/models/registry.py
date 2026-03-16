"""
Model Registry
Centralized model registration and retrieval
"""

from typing import Dict, Any, Optional
import torch.nn as nn


class ModelRegistry:
    """
    Singleton registry for managing models
    """
    
    _instance: Optional['ModelRegistry'] = None
    _models: Dict[str, nn.Module] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register(self, name: str, model: nn.Module):
        """
        Register a model
        
        Args:
            name: Model identifier
            model: PyTorch model
        """
        self._models[name] = model
    
    def get(self, name: str) -> Optional[nn.Module]:
        """
        Get registered model
        
        Args:
            name: Model identifier
        
        Returns:
            Model or None if not found
        """
        return self._models.get(name)
    
    def has(self, name: str) -> bool:
        """Check if model is registered"""
        return name in self._models
    
    def list_models(self) -> list:
        """List all registered model names"""
        return list(self._models.keys())
    
    def clear(self):
        """Clear all registered models"""
        self._models.clear()
    
    def remove(self, name: str):
        """Remove a specific model"""
        if name in self._models:
            del self._models[name]


# Global registry instance
model_registry = ModelRegistry()
