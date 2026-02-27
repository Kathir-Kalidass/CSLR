"""
RGB Stream (Enhanced Production Version)
CNN-based feature extraction from RGB frames with advanced optimizations
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, resnet18, resnet34, resnet50


class RGBStream(nn.Module):
    """
    Enhanced RGB Stream Network
    Uses ResNet18/34/50 as backbone with batch norm and dropout
    """
    
    def __init__(
        self, 
        backbone: str = "resnet18",
        feature_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
        backbone_chunk_size: int = 0,
    ):
        super().__init__()
        
        # Load backbone with proper weights
        weights = None
        if backbone == "resnet18":
            if pretrained:
                try:
                    weights = ResNet18_Weights.DEFAULT
                except Exception:
                    weights = None
            backbone_model = resnet18(weights=weights)
            backbone_dim = 512
        elif backbone == "resnet34":
            if pretrained:
                try:
                    weights = ResNet34_Weights.DEFAULT
                except Exception:
                    weights = None
            backbone_model = resnet34(weights=weights)
            backbone_dim = 512
        elif backbone == "resnet50":
            if pretrained:
                try:
                    weights = ResNet50_Weights.DEFAULT
                except Exception:
                    weights = None
            backbone_model = resnet50(weights=weights)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove final FC and avgpool layers
        self.backbone = nn.Sequential(*list(backbone_model.children())[:-1])
        
        # Advanced projection with dropout and batch norm
        self.feature_proj = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.feature_dim = feature_dim
        self.out_dim = feature_dim
        self.backbone_chunk_size = max(0, int(backbone_chunk_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (B, T, C, H, W) - Batch of frame sequences
        
        Returns:
            (B, T, feature_dim) - Frame-level features
        """
        B, T, C, H, W = x.shape
        
        # Reshape to process all frames
        x = x.view(B * T, C, H, W)

        # Extract features (optionally in chunks to reduce peak VRAM)
        if self.backbone_chunk_size > 0 and x.shape[0] > self.backbone_chunk_size:
            feature_chunks = []
            for start_idx in range(0, x.shape[0], self.backbone_chunk_size):
                end_idx = min(start_idx + self.backbone_chunk_size, x.shape[0])
                chunk_features = self.backbone(x[start_idx:end_idx]).flatten(1)
                feature_chunks.append(chunk_features)
            features = torch.cat(feature_chunks, dim=0)
        else:
            features = self.backbone(x).flatten(1)  # (B*T, backbone_dim)
        
        # Project to target dimension
        features = self.feature_proj(features)  # (B*T, feature_dim)
        
        # Reshape back to sequence
        features = features.view(B, T, self.feature_dim)
        
        return features
