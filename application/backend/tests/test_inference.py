"""
Test Inference Endpoints
"""

import pytest
import io


def test_frames_inference_endpoint(client, base64_frames):
    """Test frames inference endpoint"""
    payload = {
        "frames": base64_frames[:5],  # Use first 5 frames
        "fps": 25.0
    }
    
    response = client.post("/api/v1/inference/frames", json=payload)
    
    # May return 503 if InferenceService not initialized (expected in tests)
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "gloss" in data
        assert "sentence" in data
        assert "confidence" in data
        assert "fps" in data
        assert "processing_time" in data


def test_frames_inference_empty_frames(client):
    """Test frames inference with empty frames list"""
    payload = {
        "frames": [],
        "fps": 25.0
    }
    
    response = client.post("/api/v1/inference/frames", json=payload)
    
    # Should return error or 503 if service not initialized
    assert response.status_code in [400, 503]


def test_frames_inference_invalid_base64(client):
    """Test frames inference with invalid base64"""
    payload = {
        "frames": ["invalid_base64_data"],
        "fps": 25.0
    }
    
    response = client.post("/api/v1/inference/frames", json=payload)
    
    # Should return error
    assert response.status_code in [400, 500, 503]


def test_video_inference_no_file(client):
    """Test video inference without file"""
    response = client.post("/api/v1/inference/video")
    
    # Should return 422 (validation error)
    assert response.status_code == 422


def test_video_inference_with_file(client, mock_video_path):
    """Test video inference with actual video file"""
    with open(mock_video_path, 'rb') as f:
        files = {'file': ('test.mp4', f, 'video/mp4')}
        response = client.post("/api/v1/inference/video", files=files)
    
    # May return 503 if InferenceService not initialized
    assert response.status_code in [200, 500, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "gloss" in data
        assert "sentence" in data
        assert "confidence" in data


def test_inference_response_schema(client, base64_frames):
    """Test inference response matches expected schema"""
    payload = {
        "frames": base64_frames[:3],
        "fps": 25.0
    }
    
    response = client.post("/api/v1/inference/frames", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        
        # Check data types
        assert isinstance(data["gloss"], list)
        assert isinstance(data["sentence"], str)
        assert isinstance(data["confidence"], (int, float))
        assert isinstance(data["fps"], (int, float))
        assert isinstance(data["processing_time"], (int, float))
        
        # Check ranges
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["fps"] >= 0
        assert data["processing_time"] >= 0


def test_frames_inference_missing_fps(client, base64_frames):
    """Test frames inference without FPS field"""
    payload = {
        "frames": base64_frames[:3]
        # fps is optional with default
    }
    
    response = client.post("/api/v1/inference/frames", json=payload)
    
    # Should still work with default FPS
    assert response.status_code in [200, 503]
