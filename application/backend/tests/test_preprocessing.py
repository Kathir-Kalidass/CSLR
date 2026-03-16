"""
Test Preprocessing Pipeline
"""

import pytest
import numpy as np
import tempfile
import os


def test_video_loader_load_video(mock_video_path):
    """Test video loading from file"""
    from app.pipeline.module1_preprocessing.video_loader import VideoLoader
    
    loader = VideoLoader(target_fps=25)
    frames, fps = loader.load_video(mock_video_path)
    
    assert len(frames) > 0
    assert fps > 0
    assert isinstance(frames, list)
    assert isinstance(frames[0], np.ndarray)


def test_video_loader_invalid_path():
    """Test video loading with invalid path"""
    from app.pipeline.module1_preprocessing.video_loader import VideoLoader
    
    loader = VideoLoader()
    
    with pytest.raises(ValueError):
        loader.load_video("/non/existent/video.mp4")


def test_video_loader_from_bytes(mock_video_path):
    """Test video loading from bytes"""
    from app.pipeline.module1_preprocessing.video_loader import VideoLoader
    
    # Read video file as bytes
    with open(mock_video_path, 'rb') as f:
        video_bytes = f.read()
    
    loader = VideoLoader()
    frames, fps = loader.load_from_bytes(video_bytes)
    
    assert len(frames) > 0
    assert fps > 0


def test_frame_sampler_sampling():
    """Test frame sampling"""
    from app.pipeline.module1_preprocessing.frame_sampler import FrameSampler
    
    sampler = FrameSampler(target_fps=25)
    
    # Create mock frames
    frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(100)]
    original_fps = 30.0
    
    sampled = sampler.sample(frames, original_fps)
    
    assert len(sampled) > 0
    assert len(sampled) <= len(frames)


def test_frame_sampler_uniform_sampling():
    """Test uniform sampling mode"""
    from app.pipeline.module1_preprocessing.frame_sampler import FrameSampler
    
    sampler = FrameSampler(target_length=10)
    
    frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(50)]
    
    sampled = sampler.uniform_sample(frames, target_length=10)
    
    assert len(sampled) == 10


def test_pose_extractor_initialization():
    """Test pose extractor initialization"""
    from app.pipeline.module1_preprocessing.pose_extractor import PoseExtractor
    
    extractor = PoseExtractor()
    
    assert extractor is not None
    assert hasattr(extractor, 'extract')


def test_pose_extractor_extract(mock_frames):
    """Test pose extraction from frames"""
    from app.pipeline.module1_preprocessing.pose_extractor import PoseExtractor
    
    extractor = PoseExtractor()
    
    # Extract from first frame
    pose_data = extractor.extract(mock_frames[0])
    
    # Should return pose landmarks or None
    assert pose_data is None or isinstance(pose_data, (np.ndarray, dict))


def test_normalization_normalize():
    """Test image normalization"""
    from app.pipeline.module1_preprocessing.normalization import ImageNormalizer
    
    normalizer = ImageNormalizer()
    
    # Create random image
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    normalized = normalizer.normalize(image)
    
    assert normalized.shape == (3, 224, 224)  # CHW format
    assert normalized.dtype == np.float32


def test_normalization_batch():
    """Test batch normalization"""
    from app.pipeline.module1_preprocessing.normalization import ImageNormalizer
    
    normalizer = ImageNormalizer()
    
    # Create batch of images
    images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(5)]
    
    normalized = normalizer.normalize_batch(images)
    
    assert normalized.shape[0] == 5  # Batch size
    assert normalized.shape[1] == 3  # Channels
