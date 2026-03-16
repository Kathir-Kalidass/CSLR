"""
Checkpoint Management System
Handles save/load with automatic old checkpoint cleanup
"""

import os
import queue
from pathlib import Path
from typing import Dict, Optional

import torch

from app.core.logging import logger


class CheckpointManager:
    """
    Manages model checkpoints with automatic cleanup
    """
    
    def __init__(self, save_dir: str, keep_last: int = 5):
        """
        Args:
            save_dir: Directory to save checkpoints
            keep_last: Number of recent checkpoints to keep (older ones deleted)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_last = keep_last
        self.checkpoint_queue = queue.Queue(maxsize=keep_last)
        
        logger.info(f"CheckpointManager: {save_dir}, keep_last={keep_last}")
    
    def save(
        self,
        checkpoint: Dict,
        filename: str,
        is_best: bool = False,
    ) -> Path:
        """
        Save checkpoint and manage queue
        
        Args:
            checkpoint: State dict to save
            filename: Checkpoint filename
            is_best: If True, also save as best.pth
            
        Returns:
            Path to saved checkpoint
        """
        ckpt_path = self.save_dir / filename
        torch.save(checkpoint, ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path}")
        
        # Add to queue and cleanup
        if self.checkpoint_queue.full():
            old_ckpt = self.checkpoint_queue.get()
            if old_ckpt.exists():
                old_ckpt.unlink()
                logger.info(f"Deleted old checkpoint: {old_ckpt}")
        
        self.checkpoint_queue.put(ckpt_path)
        
        # Save best
        if is_best:
            best_path = self.save_dir / "best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Best checkpoint saved: {best_path}")
        
        return ckpt_path
    
    def load(
        self,
        filename: str = "best.pth",
        device: str = "cpu",
    ) -> Optional[Dict]:
        """
        Load checkpoint
        
        Args:
            filename: Checkpoint filename
            device: Device to load checkpoint to
            
        Returns:
            Checkpoint state dict or None if not found
        """
        ckpt_path = self.save_dir / filename
        
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            return None
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        logger.info(f"Checkpoint loaded: {ckpt_path}")
        
        return checkpoint
    
    def load_best(self, device: str = "cpu") -> Optional[Dict]:
        """Load best checkpoint"""
        return self.load("best.pth", device)
    
    def load_latest(self, device: str = "cpu") -> Optional[Dict]:
        """Load most recent checkpoint"""
        checkpoints = sorted(self.save_dir.glob("epoch_*.pth"))
        if not checkpoints:
            return None
        return self.load(checkpoints[-1].name, device)
    
    def get_checkpoint_list(self):
        """Get list of all checkpoints"""
        return sorted(self.save_dir.glob("*.pth"))
