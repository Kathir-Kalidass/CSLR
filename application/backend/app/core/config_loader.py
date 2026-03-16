"""
YAML Configuration Loader
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from app.core.logging import logger


class ConfigLoader:
    """Load and merge YAML configurations"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else None
        self.config: Dict[str, Any] = {}
    
    def load(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load YAML config file
        
        Args:
            path: Config file path
            
        Returns:
            Config dictionary
        """
        config_file = Path(path) if path else self.config_path
        
        if not config_file or not config_file.exists():
            logger.warning(f"Config file not found: {config_file}")
            return {}
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded config from {config_file}")
        self.config = config
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot-notation key"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def merge(self, override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge override config with base config"""
        def deep_merge(base, override):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        self.config = deep_merge(self.config, override)
        return self.config


# Sample training config template
TRAINING_CONFIG_TEMPLATE = """
task: CSLR
dataset: WLASL_2000

# Data configuration
data:
  train_split: train
  val_split: val
  test_split: test
  num_classes: 2000
  vocab_file: vocab.json
  
  # Video preprocessing
  video_size: [224, 224]
  num_frames: 64
  frame_rate: 25
  
  # Augmentation
  augment:
    color_jitter: 0.4
    horizontal_flip: 0.5
    temporal_crop: true

# Model configuration
model:
  name: TwoStream
  rgb_stream:
    backbone: s3d
    num_blocks: 5
    freeze_blocks: 0
    pretrained: true
  
  pose_stream:
    backbone: s3d
    num_blocks: 5
    freeze_blocks: 0
    pretrained: false
  
  fusion:
    type: lateral
    features: [c1, c2, c3]
    pose2rgb: true
    rgb2pose: true

# Training configuration
training:
  num_epochs: 100
  batch_size: 4
  num_workers: 4
  
  # Optimizer
  optimizer: adam
  learning_rate: 1.0e-3
  weight_decay: 1.0e-3
  betas: [0.9, 0.998]
  
  # Scheduler
  scheduler: cosine
  t_max: 100
  warmup_epochs: 5
  
  # Gradient clipping
  clip_grad_norm: 5.0
  
  # AMP
  use_amp: true
  
  # Checkpoint
  save_dir: checkpoints
  keep_last_ckpts: 5
  val_freq: 1

# Evaluation
eval:
  metrics: [accuracy, wer, bleu]
  beam_size: 5
"""
