"""
Pytest Configuration and Shared Fixtures
"""

import pytest
import sys
import os
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def app():
    """Create FastAPI test app"""
    from app.main import app
    return app


@pytest.fixture
def client(app):
    """Create test client"""
    from fastapi.testclient import TestClient
    return TestClient(app)


@pytest.fixture
def mock_video_path(tmp_path):
    """Create temporary video file"""
    import cv2
    import numpy as np
    
    video_path = tmp_path / "test_video.mp4"
    
    # Create dummy video
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 25.0, (224, 224))
    
    for i in range(30):  # 30 frames
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        out.write(frame)
    
    out.release()
    
    return str(video_path)


@pytest.fixture
def mock_frames():
    """Create mock frame data"""
    import numpy as np
    frames = []
    for i in range(10):
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        frames.append(frame)
    return frames


@pytest.fixture
def base64_frames(mock_frames):
    """Create base64-encoded frames"""
    import cv2
    import base64
    
    b64_frames = []
    for frame in mock_frames:
        _, buffer = cv2.imencode('.jpg', frame)
        b64 = base64.b64encode(buffer).decode('utf-8')
        b64_frames.append(b64)
    
    return b64_frames


@pytest.fixture
def sample_vocab():
    """Sample vocabulary"""
    return ["<blank>", "HELLO", "WORLD", "THANK", "YOU", "PLEASE"]


@pytest.fixture
def sample_config():
    """Sample configuration"""
    return {
        "device": "cpu",  # Use CPU for tests
        "batch_size": 1,
        "clip_length": 16,
        "target_fps": 25,
        "feature_dim": 512
    }


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create temporary test data directory"""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def api_key():
    """Sample API key for testing"""
    return "test_api_key_12345"


@pytest.fixture
def auth_headers(api_key):
    """Authentication headers"""
    return {"X-API-Key": api_key}
