"""
Training Engine
Complete training loop with validation and checkpointing
"""

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from app.core.logging import logger
from app.models.model_manager import ModelManager
from app.monitoring.performance_tracker import PerformanceTracker


class CSLRTrainer:
    """
    CSLR Training Engine with AMP, checkpointing, and validation
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: str = "checkpoints",
        device: str = "cuda",
        use_amp: bool = True,
    ):
        self.model_manager = model_manager
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_score = 0.0
        
        # Performance tracking
        self.tracker = PerformanceTracker()
        
        # AMP scaler
        self.scaler = GradScaler() if use_amp else None
        
        logger.info(f"Trainer initialized: device={device}, AMP={use_amp}")

    def _get_training_model(self) -> nn.Module:
        """
        Return the trainable model attached to model_manager.
        """
        model = getattr(self.model_manager, "model", None)
        if model is None:
            raise AttributeError("ModelManager must expose a `model` attribute for training.")
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expected `model` to be nn.Module, got {type(model).__name__}.")
        return model
    
    def build_optimizer(
        self,
        optimizer_type: str = "adam",
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.998),
    ) -> torch.optim.Optimizer:
        """Build optimizer"""
        model = self._get_training_model()

        # Get trainable parameters
        params = [p for p in model.parameters() if p.requires_grad]
        
        if optimizer_type.lower() == "adam":
            optimizer = Adam(params, lr=lr, weight_decay=weight_decay, betas=betas)
        elif optimizer_type.lower() == "adamw":
            optimizer = AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        logger.info(f"Optimizer: {optimizer_type}, LR={lr}, WD={weight_decay}")
        return optimizer
    
    def build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_type: str = "cosine",
        num_epochs: int = 100,
        step_size: int = 30,
        gamma: float = 0.1,
    ):
        """Build learning rate scheduler"""
        
        if scheduler_type.lower() == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        elif scheduler_type.lower() == "step":
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
        
        logger.info(f"Scheduler: {scheduler_type}")
        return scheduler
    
    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> Dict[str, float]:
        """Train one epoch"""
        model = self._get_training_model()
        model.train()
        
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            rgb_videos = batch['rgb'].to(self.device)
            pose_keypoints = batch['pose'].to(self.device)
            labels = batch['labels'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            
            # Forward pass with AMP
            optimizer.zero_grad()
            
            if self.use_amp:
                scaler = self.scaler
                if scaler is None:
                    raise RuntimeError("AMP is enabled but GradScaler is not initialized.")
                with autocast():
                    outputs = model(
                        rgb_videos=rgb_videos,
                        pose_keypoints=pose_keypoints,
                        labels=labels,
                        lengths=lengths,
                    )
                    loss = outputs['loss']
                
                # Backward with scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(
                    rgb_videos=rgb_videos,
                    pose_keypoints=pose_keypoints,
                    labels=labels,
                    lengths=lengths,
                )
                loss = outputs['loss']
                
                loss.backward()
                optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            if 'accuracy' in outputs:
                epoch_acc += outputs['accuracy']
            
            self.global_step += 1
            
            # Log progress
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Epoch [{epoch}/{num_batches}] "
                    f"Batch [{batch_idx+1}/{num_batches}] "
                    f"Loss: {loss.item():.4f}"
                )
        
        # Compute averages
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches if epoch_acc > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
        }
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate model"""
        model = self._get_training_model()
        model.eval()
        
        val_loss = 0.0
        val_acc = 0.0
        num_batches = len(self.val_loader)
        
        for batch in self.val_loader:
            rgb_videos = batch['rgb'].to(self.device)
            pose_keypoints = batch['pose'].to(self.device)
            labels = batch['labels'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            
            outputs = model(
                rgb_videos=rgb_videos,
                pose_keypoints=pose_keypoints,
                labels=labels,
                lengths=lengths,
            )
            
            val_loss += outputs['loss'].item()
            if 'accuracy' in outputs:
                val_acc += outputs['accuracy']
        
        avg_loss = val_loss / num_batches
        avg_acc = val_acc / num_batches if val_acc > 0 else 0.0
        
        logger.info(f"Validation - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
        
        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
        }
    
    def save_checkpoint(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler,
        score: float,
        is_best: bool = False,
    ):
        """Save training checkpoint"""
        model = self._get_training_model()
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_score': self.best_score,
            'current_score': score,
        }
        
        # Save epoch checkpoint
        ckpt_path = self.save_dir / f"epoch_{epoch:03d}.pth"
        torch.save(checkpoint, ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / "best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Best checkpoint saved: {best_path}")
    
    def train(
        self,
        num_epochs: int = 100,
        optimizer_cfg: Optional[Dict] = None,
        scheduler_cfg: Optional[Dict] = None,
        val_freq: int = 1,
    ):
        """
        Main training loop
        
        Args:
            num_epochs: Number of training epochs
            optimizer_cfg: Optimizer configuration
            scheduler_cfg: Scheduler configuration  
            val_freq: Validation frequency (epochs)
        """
        
        # Build optimizer and scheduler
        optimizer_cfg = optimizer_cfg or {}
        scheduler_cfg = scheduler_cfg or {}
        
        optimizer = self.build_optimizer(**optimizer_cfg)
        scheduler = self.build_scheduler(optimizer, num_epochs=num_epochs, **scheduler_cfg)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            start_time = time.time()
            val_metrics: Dict[str, float] = {'loss': 0.0, 'accuracy': 0.0}
            
            # Train
            train_metrics = self.train_epoch(optimizer, epoch)
            
            # Validate
            if epoch % val_freq == 0:
                val_metrics = self.validate(epoch)
                
                # Check if best
                score = val_metrics['accuracy']
                is_best = score > self.best_score
                if is_best:
                    self.best_score = score
                    logger.info(f"New best score: {score:.4f}")
                
                # Save checkpoint
                self.save_checkpoint(epoch, optimizer, scheduler, score, is_best)
            
            # Step scheduler
            scheduler.step()
            
            # Log epoch summary
            epoch_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch} completed in {epoch_time:.2f}s - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics.get('accuracy', 0):.4f}"
            )
        
        logger.info("Training completed!")
