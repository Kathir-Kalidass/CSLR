"""
Optimizer and Scheduler Builders
Production-ready with gradient clipping support
"""

from typing import Callable, Optional

import torch
from torch import nn
from torch.optim import Optimizer, Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR

from app.core.logging import logger


def build_optimizer(
    model: nn.Module,
    optimizer_type: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    betas: tuple = (0.9, 0.998),
    momentum: float = 0.9,
) -> Optimizer:
    """
    Build optimizer from config
    
    Args:
        model: PyTorch model
        optimizer_type: adam, adamw, sgd
        lr: Learning rate
        weight_decay: Weight decay (L2 regularization)
        betas: Adam betas
        momentum: SGD momentum
        
    Returns:
        Optimizer instance
    """
    params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_type.lower() == "adam":
        optimizer = Adam(
            params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
    elif optimizer_type.lower() == "adamw":
        optimizer = AdamW(
            params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
    elif optimizer_type.lower() == "sgd":
        optimizer = SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    logger.info(
        f"Optimizer built: {optimizer_type}, LR={lr}, "
        f"WD={weight_decay}, Params={len(params)}"
    )
    
    return optimizer


def build_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    num_epochs: int = 100,
    warmup_epochs: int = 0,
    step_size: int = 30,
    gamma: float = 0.1,
    milestones: Optional[list] = None,
):
    """
    Build learning rate scheduler
    
    Args:
        optimizer: Optimizer instance
        scheduler_type: cosine, step, multistep
        num_epochs: Total training epochs
        warmup_epochs: Warmup epochs (not implemented)
        step_size: Step scheduler step size
        gamma: LR decay factor
        milestones: MultiStep milestones
        
    Returns:
        Scheduler instance
    """
    if scheduler_type.lower() == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=0,
        )
    elif scheduler_type.lower() == "step":
        scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    elif scheduler_type.lower() == "multistep":
        if milestones is None:
            milestones = [num_epochs // 2, num_epochs * 3 // 4]
        scheduler = MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    logger.info(f"Scheduler built: {scheduler_type}")
    
    return scheduler


def build_gradient_clipper(
    clip_type: str = "norm",
    clip_value: float = 1.0,
) -> Optional[Callable]:
    """
    Build gradient clipping function
    
    Args:
        clip_type: norm or value
        clip_value: Clipping threshold
        
    Returns:
        Clipping function or None
    """
    if clip_type == "norm":
        return lambda params: nn.utils.clip_grad_norm_(
            params,
            max_norm=clip_value,
        )
    elif clip_type == "value":
        return lambda params: nn.utils.clip_grad_value_(
            params,
            clip_value=clip_value,
        )
    else:
        return None
