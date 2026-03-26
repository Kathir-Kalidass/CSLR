"""
Two-Stream Network with Lateral Connections
RGB + Pose streams with cross-modal fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.models.backbones.s3d import S3DBackbone


class LateralConnection(nn.Module):
    """
    Lateral connection between RGB and Pose streams
    Enables cross-modal feature enhancement
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: str = "pose2rgb",
    ):
        super().__init__()
        
        self.mode = mode
        
        # 1x1 conv for channel matching
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, source, target):
        """
        Args:
            source: Features to transfer from
            target: Features to enhance
            
        Returns:
            (source, enhanced_target)
        """
        # Match spatial dimensions
        if source.shape[-2:] != target.shape[-2:]:
            source = F.interpolate(
                source,
                size=target.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        
        # Transform and add
        lateral = self.conv(source)
        lateral = self.bn(lateral)
        lateral = self.relu(lateral)
        
        enhanced = target + lateral
        
        return source, enhanced


class TwoStreamNetwork(nn.Module):
    """
    Two-Stream Network for CSLR
    RGB + Pose with lateral connections
    """
    
    def __init__(
        self,
        num_classes: int,
        rgb_channels: int = 3,
        pose_channels: int = 2,  # x, y coordinates
        num_blocks: int = 5,
        freeze_blocks: tuple = (0, 0),
        use_lateral: tuple = (True, True),  # (pose2rgb, rgb2pose)
        fusion_features: list | None = None,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_lateral = use_lateral
        self.fusion_features = fusion_features or ['c1', 'c2', 'c3']
        
        # RGB stream
        self.rgb_stream = S3DBackbone(
            in_channels=rgb_channels,
            num_blocks=num_blocks,
            freeze_blocks=freeze_blocks[0],
        )
        
        # Pose stream
        self.pose_stream = S3DBackbone(
            in_channels=pose_channels,
            num_blocks=num_blocks,
            freeze_blocks=freeze_blocks[1],
        )
        
        # Lateral connections
        out_channels = [64, 192, 480, 832, 1024]
        fuse_indices = self._get_fusion_indices()
        
        if use_lateral[0]:  # pose2rgb
            self.pose2rgb_lateral = nn.ModuleList([
                LateralConnection(
                    out_channels[i],
                    out_channels[i],
                    mode="pose2rgb"
                )
                for i in fuse_indices
            ])
        
        if use_lateral[1]:  # rgb2pose
            self.rgb2pose_lateral = nn.ModuleList([
                LateralConnection(
                    out_channels[i],
                    out_channels[i],
                    mode="rgb2pose"
                )
                for i in fuse_indices
            ])
        
        # Classification heads
        total_dim = sum(out_channels[i] for i in range(num_blocks))
        self.classifier = nn.Sequential(
            nn.Linear(total_dim * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )
    
    def _get_fusion_indices(self):
        """Get indices where lateral fusion occurs"""
        feature_map = {
            'c1': 0,
            'c2': 1,
            'c3': 2,
            'c4': 3,
            'c5': 4,
        }
        return [feature_map[f] for f in self.fusion_features if f in feature_map]
    
    def forward(
        self,
        rgb_videos,
        pose_keypoints,
        labels=None,
        lengths=None,
    ):
        """
        Args:
            rgb_videos: (B, C, T, H, W)
            pose_keypoints: (B, T, N, D) -> converted to (B, D, T, H, W)
            labels: (B, L) for CTC loss
            lengths: (B,) label lengths
            
        Returns:
            Dict with logits, loss, features
        """
        B = rgb_videos.size(0)
        
        # Convert pose to spatial format if needed
        if pose_keypoints.dim() == 4:  # (B, T, N, D)
            # Simple conversion: treat as image
            T, N, D = pose_keypoints.shape[1:]
            H = W = int(N ** 0.5) if N > 1 else 1
            pose_keypoints = pose_keypoints.permute(0, 3, 1, 2).unsqueeze(-1)
            pose_keypoints = F.interpolate(
                pose_keypoints,
                size=(T, H, W),
                mode='trilinear',
                align_corners=False,
            )
        
        # Extract features
        rgb_features = self.rgb_stream(rgb_videos)
        pose_features = self.pose_stream(pose_keypoints)
        
        # Apply lateral connections
        fuse_indices = self._get_fusion_indices()
        
        for i, idx in enumerate(fuse_indices):
            if self.use_lateral[0]:  # pose2rgb
                pose_features[idx], rgb_features[idx] = self.pose2rgb_lateral[i](
                    pose_features[idx],
                    rgb_features[idx],
                )
            
            if self.use_lateral[1]:  # rgb2pose
                rgb_features[idx], pose_features[idx] = self.rgb2pose_lateral[i](
                    rgb_features[idx],
                    pose_features[idx],
                )
        
        # Global pooling and concatenation
        rgb_pooled = []
        pose_pooled = []
        
        for rgb_feat, pose_feat in zip(rgb_features, pose_features):
            # Spatial-temporal pooling
            rgb_p = F.adaptive_avg_pool3d(rgb_feat, (1, 1, 1))
            pose_p = F.adaptive_avg_pool3d(pose_feat, (1, 1, 1))
            
            rgb_pooled.append(rgb_p.view(B, -1))
            pose_pooled.append(pose_p.view(B, -1))
        
        # Concatenate all features
        rgb_concat = torch.cat(rgb_pooled, dim=1)
        pose_concat = torch.cat(pose_pooled, dim=1)
        fused = torch.cat([rgb_concat, pose_concat], dim=1)
        
        # Classification
        logits = self.classifier(fused)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return {
            'logits': logits,
            'loss': loss,
            'rgb_features': rgb_features,
            'pose_features': pose_features,
        }
