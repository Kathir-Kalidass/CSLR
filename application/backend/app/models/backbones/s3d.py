"""
S3D Backbone from I3D Architecture
Separable 3D convolutions for efficient video processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SepConv3d(nn.Module):
    """Separable 3D Convolution (Spatial + Temporal)"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        
        # Spatial convolution (1, k, k)
        self.spatial_conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size, kernel_size),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False,
        )
        
        # Temporal convolution (k, 1, 1)
        self.temporal_conv = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(kernel_size, 1, 1),
            stride=(stride, 1, 1),
            padding=(padding, 0, 0),
            bias=False,
        )
        
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BasicBlock3D(nn.Module):
    """3D ResNet Basic Block"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = SepConv3d(in_channels, out_channels, 3, stride, 1)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Downsample if needed
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class S3DBackbone(nn.Module):
    """
    S3D Backbone for Video Feature Extraction
    Based on I3D with separable convolutions
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_blocks: int = 5,
        freeze_blocks: int = 0,
    ):
        super().__init__()
        
        self.num_blocks = num_blocks
        self.freeze_blocks = freeze_blocks
        
        # Block configurations: (out_channels, num_layers, stride)
        configs = [
            (64, 1, 1),      # Block 0
            (192, 2, 2),     # Block 1
            (480, 3, 2),     # Block 2
            (832, 4, 2),     # Block 3
            (1024, 2, 2),    # Block 4
        ]
        
        # Build blocks
        self.blocks = nn.ModuleList()
        current_channels = in_channels
        
        for i in range(min(num_blocks, len(configs))):
            out_ch, num_layers, stride = configs[i]
            
            layers = []
            for j in range(num_layers):
                layers.append(
                    BasicBlock3D(
                        current_channels,
                        out_ch,
                        stride if j == 0 else 1,
                    )
                )
                current_channels = out_ch
            
            self.blocks.append(nn.Sequential(*layers))
        
        # Freeze blocks
        self._freeze_blocks()
    
    def _freeze_blocks(self):
        """Freeze first N blocks"""
        for i in range(self.freeze_blocks):
            if i < len(self.blocks):
                for param in self.blocks[i].parameters():
                    param.requires_grad = False
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W)
            
        Returns:
            List of features from each block
        """
        features = []
        
        for block in self.blocks:
            x = block(x)
            features.append(x)
        
        return features
    
    @property
    def out_channels(self):
        """Output channels of each block"""
        return [64, 192, 480, 832, 1024][:self.num_blocks]
