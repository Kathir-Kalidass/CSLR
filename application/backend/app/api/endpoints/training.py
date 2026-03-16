"""
Training API Endpoints
Manage training jobs via REST API
"""

from typing import Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.core.logging import logger
from app.training.trainer import CSLRTrainer
from app.training.checkpoint_manager import CheckpointManager
from app.data.video_dataset import CSLRVideoDataset, collate_fn
from app.models.two_stream import TwoStreamNetwork

from torch.utils.data import DataLoader

router = APIRouter(prefix="/training", tags=["training"])


class TrainingConfig(BaseModel):
    """Training configuration"""
    data_dir: str
    num_epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    scheduler: str = "cosine"
    optimizer: str = "adam"
    num_classes: int = 2000
    val_freq: int = 1
    save_dir: str = "checkpoints"


class TrainingStatus(BaseModel):
    """Training job status"""
    status: str
    current_epoch: int
    total_epochs: int
    best_score: float
    message: str


# Global training state
training_state = {
    "active": False,
    "trainer": None,
    "current_epoch": 0,
    "total_epochs": 0,
    "best_score": 0.0,
}


@router.post("/start")
async def start_training(
    config: TrainingConfig,
    background_tasks: BackgroundTasks,
):
    """
    Start training job
    
    Args:
        config: Training configuration
        
    Returns:
        Job status
    """
    if training_state["active"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    logger.info(f"Starting training: {config.dict()}")
    
    # Create model
    model = TwoStreamNetwork(
        num_classes=config.num_classes,
        num_blocks=5,
        freeze_blocks=(0, 0),
        use_lateral=(True, True),
    )
    
    # Create datasets
    train_dataset = CSLRVideoDataset(
        data_dir=config.data_dir,
        split="train",
        augment=True,
    )
    
    val_dataset = CSLRVideoDataset(
        data_dir=config.data_dir,
        split="val",
        augment=False,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    
    # Create trainer
    from app.models.model_manager import ModelManager
    model_manager = ModelManager()
    
    # Wrap model for training
    class TrainingModelManager:
        def __init__(self, model):
            self.model = model
    
    training_model_manager = TrainingModelManager(model.cuda())
    
    trainer = CSLRTrainer(
        model_manager=model_manager,
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=config.save_dir,
        use_amp=True,
    )
    
    # Update state
    training_state["trainer"] = trainer
    training_state["total_epochs"] = config.num_epochs
    training_state["active"] = True
    
    # Start training in background
    def train_job():
        try:
            trainer.train(
                num_epochs=config.num_epochs,
                optimizer_cfg={
                    "optimizer_type": config.optimizer,
                    "lr": config.learning_rate,
                    "weight_decay": config.weight_decay,
                },
                scheduler_cfg={
                    "scheduler_type": config.scheduler,
                },
                val_freq=config.val_freq,
            )
            training_state["active"] = False
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            training_state["active"] = False
            raise
    
    background_tasks.add_task(train_job)
    
    return {
        "status": "started",
        "message": f"Training started for {config.num_epochs} epochs",
    }


@router.get("/status", response_model=TrainingStatus)
async def get_training_status():
    """Get current training status"""
    
    trainer = training_state.get("trainer")
    
    return TrainingStatus(
        status="running" if training_state["active"] else "idle",
        current_epoch=trainer.current_epoch if trainer else 0,
        total_epochs=training_state["total_epochs"],
        best_score=trainer.best_score if trainer else 0.0,
        message="Training in progress" if training_state["active"] else "No active training",
    )


@router.post("/stop")
async def stop_training():
    """Stop current training job"""
    
    if not training_state["active"]:
        raise HTTPException(status_code=400, detail="No active training")
    
    training_state["active"] = False
    logger.info("Training stopped by user")
    
    return {"status": "stopped", "message": "Training will stop after current epoch"}


@router.get("/checkpoints")
async def list_checkpoints(save_dir: str = "checkpoints"):
    """List all saved checkpoints"""
    
    ckpt_manager = CheckpointManager(save_dir)
    checkpoints = ckpt_manager.get_checkpoint_list()
    
    return {
        "checkpoints": [str(ckpt) for ckpt in checkpoints],
        "count": len(checkpoints),
    }


@router.post("/load-checkpoint")
async def load_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
):
    """Load a checkpoint"""
    
    import torch
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        return {
            "status": "success",
            "epoch": checkpoint.get("epoch", 0),
            "best_score": checkpoint.get("best_score", 0.0),
            "message": f"Checkpoint loaded from {checkpoint_path}",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load checkpoint: {e}")
