"""
Model Loading Utilities
Loads pre-trained models from checkpoints with automatic checkpoint detection
"""

import torch
import os
from typing import Dict, Any, Optional
from pathlib import Path
from app.core.config import settings
from app.core.logging import logger


def find_latest_checkpoint(checkpoint_dir: str, model_name: str) -> Optional[str]:
    """
    Find the latest checkpoint for a given model
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        model_name: Name of the model (e.g., 'rgb', 'pose', 'fusion', 'sequence')
    
    Returns:
        Path to latest checkpoint or None
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None
    
    # Find all checkpoints for this model
    pattern = f"{model_name}_*.pth"
    checkpoints = list(checkpoint_path.glob(pattern))
    
    if not checkpoints:
        # Try 'best' checkpoint
        best_checkpoint = checkpoint_path / f"{model_name}_best.pth"
        if best_checkpoint.exists():
            return str(best_checkpoint)
        return None
    
    # Sort by modification time, get most recent
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest)


def load_model_checkpoint(
    model_path: str,
    device: str = "cuda",
    map_location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint
    
    Args:
        model_path: Path to checkpoint file
        device: Target device
        map_location: Optional map location for loading
    
    Returns:
        Checkpoint dictionary
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    if map_location is None:
        map_location = device
    
    logger.info(f"Loading checkpoint from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=map_location)
    
    logger.info(f"Checkpoint loaded successfully")
    
    return checkpoint


def load_rgb_model(device: Optional[str] = None, checkpoint_path: Optional[str] = None):
    """Load RGB stream model"""
    if device is None:
        device = settings.DEVICE
    
    from app.pipeline.module2_feature.rgb_stream import RGBStream
    
    model = RGBStream(backbone="resnet18", feature_dim=512)
    
    # Load pretrained weights if available
    if checkpoint_path is None:
        checkpoint_dir = getattr(settings, 'CHECKPOINT_DIR', 'checkpoints')
        checkpoint_path = find_latest_checkpoint(checkpoint_dir, 'rgb')
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading RGB weights from {checkpoint_path}")
        checkpoint = load_model_checkpoint(checkpoint_path, device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info("RGB weights loaded successfully")
    else:
        logger.warning("No RGB checkpoint found, using random initialization")
    
    model.to(device)
    model.eval()
    
    logger.info("RGB model loaded")
    return model


def load_pose_model(device: Optional[str] = None, checkpoint_path: Optional[str] = None):
    """Load Pose stream model"""
    if device is None:
        device = settings.DEVICE
    
    from app.pipeline.module2_feature.pose_stream import PoseStream
    
    model = PoseStream(input_dim=258, feature_dim=512)
    
    # Load pretrained weights if available
    if checkpoint_path is None:
        checkpoint_dir = getattr(settings, 'CHECKPOINT_DIR', 'checkpoints')
        checkpoint_path = find_latest_checkpoint(checkpoint_dir, 'pose')
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading Pose weights from {checkpoint_path}")
        checkpoint = load_model_checkpoint(checkpoint_path, device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info("Pose weights loaded successfully")
    else:
        logger.warning("No Pose checkpoint found, using random initialization")
    
    model.to(device)
    model.eval()
    
    logger.info("Pose model loaded")
    return model


def load_fusion_model(device: Optional[str] = None, checkpoint_path: Optional[str] = None):
    """Load Feature fusion model"""
    if device is None:
        device = settings.DEVICE
    
    from app.pipeline.module2_feature.fusion import FeatureFusion
    
    model = FeatureFusion(
        rgb_dim=512,
        pose_dim=512,
        fusion_dim=512,
        fusion_type="concat"
    )
    
    # Load pretrained weights if available
    if checkpoint_path is None:
        checkpoint_dir = getattr(settings, 'CHECKPOINT_DIR', 'checkpoints')
        checkpoint_path = find_latest_checkpoint(checkpoint_dir, 'fusion')
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading Fusion weights from {checkpoint_path}")
        checkpoint = load_model_checkpoint(checkpoint_path, device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info("Fusion weights loaded successfully")
    else:
        logger.warning("No Fusion checkpoint found, using random initialization")
    
    model.to(device)
    model.eval()
    
    logger.info("Fusion model loaded")
    return model


def load_sequence_model(device: Optional[str] = None, vocab_size: int = 1000, checkpoint_path: Optional[str] = None):
    """Load Sequence model (BiLSTM + CTC)"""
    if device is None:
        device = settings.DEVICE
    
    from app.pipeline.module3_sequence.temporal_model import TemporalModel
    
    model = TemporalModel(
        input_dim=512,
        hidden_dim=256,
        num_layers=2,
        vocab_size=vocab_size,
        model_type="bilstm"
    )
    
    # Load pretrained weights if available
    if checkpoint_path is None:
        checkpoint_dir = getattr(settings, 'CHECKPOINT_DIR', 'checkpoints')
        checkpoint_path = find_latest_checkpoint(checkpoint_dir, 'sequence')
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading Sequence weights from {checkpoint_path}")
        checkpoint = load_model_checkpoint(checkpoint_path, device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info("Sequence weights loaded successfully")
    else:
        logger.warning("No Sequence checkpoint found, using random initialization")
    
    model.to(device)
    model.eval()
    
    logger.info("Sequence model loaded")
    return model


def load_all_models(device: Optional[str] = None) -> Dict[str, Any]:
    """
    Load all models
    
    Returns:
        Dictionary of loaded models
    """
    if device is None:
        device = settings.DEVICE
    
    logger.info("Loading all models...")
    
    models = {
        "rgb": load_rgb_model(device),
        "pose": load_pose_model(device),
        "fusion": load_fusion_model(device),
        "sequence": load_sequence_model(device)
    }
    
    logger.info("All models loaded successfully")
    
    return models
