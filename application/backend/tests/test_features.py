"""
Test Feature Extraction Modules
"""

import pytest
import torch
import numpy as np


def test_rgb_stream_initialization():
    """Test RGB stream model initialization"""
    from app.pipeline.module2_feature.rgb_stream import RGBStream
    
    model = RGBStream(backbone="resnet18", feature_dim=512)
    
    assert model is not None
    assert hasattr(model, 'forward')


def test_rgb_stream_forward():
    """Test RGB stream forward pass"""
    from app.pipeline.module2_feature.rgb_stream import RGBStream
    
    model = RGBStream(backbone="resnet18", feature_dim=512)
    model.eval()
    
    # Create dummy input (B, C, T, H, W)
    x = torch.randn(1, 3, 8, 224, 224)
    
    with torch.no_grad():
        output = model(x)
    
    # Output should be (B, T, feature_dim)
    assert output.shape[0] == 1
    assert output.shape[2] == 512


def test_pose_stream_initialization():
    """Test Pose stream model initialization"""
    from app.pipeline.module2_feature.pose_stream import PoseStream
    
    model = PoseStream(input_dim=258, feature_dim=512)
    
    assert model is not None


def test_pose_stream_forward():
    """Test Pose stream forward pass"""
    from app.pipeline.module2_feature.pose_stream import PoseStream
    
    model = PoseStream(input_dim=258, feature_dim=512)
    model.eval()
    
    # Create dummy pose input (B, T, 258)
    x = torch.randn(1, 8, 258)
    
    with torch.no_grad():
        output = model(x)
    
    # Output should be (B, T, 512)
    assert output.shape == (1, 8, 512)


def test_feature_fusion_concat():
    """Test feature fusion with concatenation"""
    from app.pipeline.module2_feature.fusion import FeatureFusion
    
    model = FeatureFusion(
        rgb_dim=512,
        pose_dim=512,
        fusion_dim=512,
        fusion_type="concat"
    )
    
    rgb_features = torch.randn(1, 8, 512)
    pose_features = torch.randn(1, 8, 512)
    
    with torch.no_grad():
        fused = model(rgb_features, pose_features)
    
    assert fused.shape[0] == 1
    assert fused.shape[1] == 8


def test_feature_fusion_gated_attention():
    """Test feature fusion with gated attention"""
    from app.pipeline.module2_feature.fusion import FeatureFusion
    
    model = FeatureFusion(
        rgb_dim=512,
        pose_dim=512,
        fusion_dim=512,
        fusion_type="gated_attention"
    )
    
    rgb_features = torch.randn(1, 8, 512)
    pose_features = torch.randn(1, 8, 512)
    
    with torch.no_grad():
        fused = model(rgb_features, pose_features)
    
    assert fused.shape == (1, 8, 512)


def test_attention_module():
    """Test attention mechanism"""
    from app.pipeline.module2_feature.attention import MultiHeadAttention
    
    attention = MultiHeadAttention(embed_dim=512, num_heads=8)
    
    query = torch.randn(1, 8, 512)
    key = torch.randn(1, 8, 512)
    value = torch.randn(1, 8, 512)
    
    with torch.no_grad():
        output, weights = attention(query, key, value)
    
    assert output.shape == query.shape
    assert weights.shape[0] == 1  # Batch
