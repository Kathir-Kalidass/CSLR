"""
Pose Stream (Enhanced Production Version)
MLP-based feature extraction from pose keypoints with batch normalization
"""

import torch
import torch.nn as nn


class PoseStream(nn.Module):
    """
    Enhanced Pose Stream Network
    Multi-layer MLP for pose embeddings with batch norm
    """
    
    def __init__(
        self,
        input_dim: int = 258,  # MediaPipe keypoints (33 pose + 21*2 hands) * 2 coords = 150, or with visibility = 258
        hidden_dims: list = [512, 256],
        feature_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers with batch normalization
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output projection
        layers.append(nn.Linear(prev_dim, feature_dim))
        
        self.mlp = nn.Sequential(*layers)
        self.feature_dim = feature_dim
        self.out_dim = feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (B, T, input_dim) or (B, T, N, D) - Pose keypoint sequences
        
        Returns:
            (B, T, feature_dim) - Pose features
        """
        # Handle both (B, T, N, D) and (B, T, input_dim) formats
        if x.dim() == 4:
            B, T, N, D = x.shape
            x = x.view(B, T, N * D)
        
        B, T, D = x.shape
        
        # Flatten temporal dimension
        x = x.view(B * T, D)
        
        # Extract features
        features = self.mlp(x)  # (B*T, feature_dim)
        
        # Reshape back
        features = features.view(B, T, self.feature_dim)
        
        return features
