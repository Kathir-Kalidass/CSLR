"""
Fusion Module (Enhanced Production Version)
Combines RGB and Pose features with advanced attention mechanisms
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class FeatureFusion(nn.Module):
    """
    Enhanced Feature Fusion Module
    Combines RGB and Pose streams with gated attention
    """
    
    def __init__(
        self,
        rgb_dim: int = 512,
        pose_dim: int = 512,
        fusion_dim: int = 512,
        fusion_type: str = "gated_attention"  # concat, add, attention, gated_attention
    ):
        super().__init__()
        
        self.fusion_type = fusion_type
        self.fusion_dim = fusion_dim
        
        if fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(rgb_dim + pose_dim, fusion_dim),
                nn.BatchNorm1d(fusion_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            )
        
        elif fusion_type == "add":
            assert rgb_dim == pose_dim == fusion_dim, "All dims must match for addition"
            # No learnable parameters needed
        
        elif fusion_type == "attention":
            self.attention = MultimodalAttention(rgb_dim, pose_dim, fusion_dim)
        
        elif fusion_type == "gated_attention":
            # Advanced gated attention fusion (from production model)
            self.pose_proj = nn.Linear(pose_dim, fusion_dim)
            self.rgb_gate = nn.Linear(rgb_dim, fusion_dim)
            self.pose_gate = nn.Linear(pose_dim, fusion_dim)
            self.norm = nn.LayerNorm(fusion_dim)
        
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(
        self, 
        rgb_features: torch.Tensor,
        pose_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Fuse RGB and Pose features
        
        Args:
            rgb_features: (B, T, rgb_dim)
            pose_features: (B, T, pose_dim)
        
        Returns:
            fused: (B, T, fusion_dim) - Fused features
            alpha: (B, T, fusion_dim) - RGB attention weights (optional)
            beta: (B, T, fusion_dim) - Pose attention weights (optional)
        """
        fused: torch.Tensor
        alpha: Optional[torch.Tensor] = None
        beta: Optional[torch.Tensor] = None
        
        if self.fusion_type == "concat":
            # Concatenate and project
            B, T, _ = rgb_features.shape
            combined = torch.cat([rgb_features, pose_features], dim=-1)
            combined = combined.view(B * T, -1)
            fused = self.fusion(combined)
            fused = fused.view(B, T, self.fusion_dim)
        
        elif self.fusion_type == "add":
            # Element-wise addition
            fused = rgb_features + pose_features
        
        elif self.fusion_type == "attention":
            # Attention-based fusion
            fused = self.attention(rgb_features, pose_features)
        
        elif self.fusion_type == "gated_attention":
            # Gated attention fusion (production-ready)
            pose_aligned = self.pose_proj(pose_features)
            alpha = torch.sigmoid(self.rgb_gate(rgb_features))
            beta = torch.sigmoid(self.pose_gate(pose_features))
            fused = self.norm(alpha * rgb_features + beta * pose_aligned)
        
        else:
            # Fallback to concatenation
            combined = torch.cat([rgb_features, pose_features], dim=-1)
            fused = combined
        
        return fused, alpha, beta


class MultimodalAttention(nn.Module):
    """Cross-modal attention fusion with bidirectional attention"""
    
    def __init__(self, rgb_dim: int, pose_dim: int, output_dim: int, num_heads: int = 8):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        assert self.head_dim * num_heads == output_dim, "output_dim must be divisible by num_heads"
        
        # RGB queries Pose
        self.query_rgb = nn.Linear(rgb_dim, output_dim)
        self.key_pose = nn.Linear(pose_dim, output_dim)
        self.value_pose = nn.Linear(pose_dim, output_dim)
        
        # Pose queries RGB (bidirectional)
        self.query_pose = nn.Linear(pose_dim, output_dim)
        self.key_rgb = nn.Linear(rgb_dim, output_dim)
        self.value_rgb = nn.Linear(rgb_dim, output_dim)
        
        # Final projection
        self.output_proj = nn.Linear(output_dim * 2, output_dim)
        
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(output_dim)
    
    def _multi_head_attention(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor
    ) -> torch.Tensor:
        """
        Multi-head attention mechanism
        
        Args:
            Q, K, V: (B, T, output_dim)
        
        Returns:
            (B, T, output_dim)
        """
        B, T, D = Q.shape
        
        # Reshape to multi-head: (B, num_heads, T, head_dim)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, num_heads, T, T)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Weighted values
        out = torch.matmul(attn, V)  # (B, num_heads, T, head_dim)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        
        return out
    
    def forward(self, rgb: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """
        Bidirectional cross-attention
        
        Args:
            rgb: (B, T, rgb_dim)
            pose: (B, T, pose_dim)
        
        Returns:
            (B, T, output_dim)
        """
        # RGB queries Pose
        Q_rgb = self.query_rgb(rgb)
        K_pose = self.key_pose(pose)
        V_pose = self.value_pose(pose)
        rgb_to_pose = self._multi_head_attention(Q_rgb, K_pose, V_pose)
        
        # Pose queries RGB
        Q_pose = self.query_pose(pose)
        K_rgb = self.key_rgb(rgb)
        V_rgb = self.value_rgb(rgb)
        pose_to_rgb = self._multi_head_attention(Q_pose, K_rgb, V_rgb)
        
        # Combine bidirectional attention
        combined = torch.cat([rgb_to_pose, pose_to_rgb], dim=-1)
        out = self.output_proj(combined)
        out = self.norm(out)
        
        return out
