"""
Module 2: Feature Extraction
RGB stream, Pose stream, Fusion
"""

from app.pipeline.module2_feature.rgb_stream import RGBStream
from app.pipeline.module2_feature.pose_stream import PoseStream
from app.pipeline.module2_feature.fusion import FeatureFusion
from app.pipeline.module2_feature.attention import MultiHeadAttention

__all__ = ['RGBStream', 'PoseStream', 'FeatureFusion', 'MultiHeadAttention']
