"""
Model Warmup Script
Runs initial inference to initialize CUDA and warm up models
"""

import torch
import time
import argparse
from typing import Optional
from app.models.load_model import load_all_models
from app.core.config import settings
from app.core.logging import logger


def warmup_model(model, device: str, num_runs: int = 10):
    """
    Warmup a single model
    
    Args:
        model: PyTorch model
        device: Device to use
        num_runs: Number of warmup runs
    """
    model.eval()
    model.to(device)
    
    # Create dummy input
    if hasattr(model, 'feature_dim'):
        # Sequence model
        dummy_input = torch.randn(1, 32, 512, device=device)
    else:
        # Feature extraction model
        dummy_input = torch.randn(1, 32, 3, 224, 224, device=device)
    
    # Warmup runs
    logger.info(f"Warming up model with {num_runs} runs...")
    
    with torch.no_grad():
        for i in range(num_runs):
            start = time.time()
            _ = model(dummy_input)
            elapsed = time.time() - start
            
            logger.debug(f"Run {i+1}/{num_runs}: {elapsed*1000:.2f}ms")
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    logger.info("Warmup complete!")


def warmup_all_models(device: Optional[str] = None, num_runs: int = 10):
    """
    Warmup all models
    
    Args:
        device: Device to use
        num_runs: Number of warmup runs per model
    """
    if device is None:
        device = settings.DEVICE
    
    logger.info(f"Loading models on {device}...")
    models = load_all_models(device)
    
    for name, model in models.items():
        logger.info(f"\nWarming up {name} model...")
        try:
            warmup_model(model, device, num_runs)
        except Exception as e:
            logger.error(f"Warmup failed for {name}: {e}")
    
    # GPU memory summary
    if device == "cuda" and torch.cuda.is_available():
        logger.info(f"\nGPU Memory Summary:")
        logger.info(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        logger.info(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        logger.info(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Warmup models")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cpu or cuda)"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of warmup runs"
    )
    
    args = parser.parse_args()
    
    logger.info("=== Model Warmup Script ===")
    warmup_all_models(device=args.device, num_runs=args.num_runs)
    logger.info("\n=== Warmup Complete ===")


if __name__ == "__main__":
    main()
