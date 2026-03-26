#!/usr/bin/env python3
"""
API Test Script
Test all backend endpoints
"""

import requests
import json
import base64
import time
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000/api/v1"
API_KEY = None  # Set if authentication enabled


def test_health():
    """Test health endpoint"""
    print("\n" + "=" * 60)
    print("Testing Health Endpoint")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_system_info():
    """Test system info endpoint"""
    print("\n" + "=" * 60)
    print("Testing System Info Endpoint")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/health/system")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_model_status():
    """Test model status endpoint"""
    print("\n" + "=" * 60)
    print("Testing Model Status Endpoint")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/health/models")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_video_inference(video_path: str):
    """Test video inference endpoint"""
    print("\n" + "=" * 60)
    print("Testing Video Inference Endpoint")
    print("=" * 60)
    
    if not Path(video_path).exists():
        print(f"Video file not found: {video_path}")
        return False
    
    headers = {}
    if API_KEY:
        headers['X-API-Key'] = API_KEY
    
    with open(video_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            f"{BASE_URL}/inference/video",
            files=files,
            headers=headers
        )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return True
    else:
        print(f"Error: {response.text}")
        return False


def test_frames_inference():
    """Test frames inference endpoint"""
    print("\n" + "=" * 60)
    print("Testing Frames Inference Endpoint")
    print("=" * 60)
    
    # Create dummy image frames
    import numpy as np
    import cv2
    
    frames = []
    for i in range(5):
        # Create random frame
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        # Base64 encode
        b64 = base64.b64encode(buffer).decode('utf-8')
        frames.append(b64)
    
    payload = {
        "frames": frames,
        "fps": 25.0
    }
    
    headers = {'Content-Type': 'application/json'}
    if API_KEY:
        headers['X-API-Key'] = API_KEY
    
    response = requests.post(
        f"{BASE_URL}/inference/frames",
        json=payload,
        headers=headers
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return True
    else:
        print(f"Error: {response.text}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("CSLR Backend API Tests")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print(f"API Key: {'Set' if API_KEY else 'Not set'}")
    
    results = {}
    
    # Run tests
    results['health'] = test_health()
    results['system_info'] = test_system_info()
    results['model_status'] = test_model_status()
    results['frames_inference'] = test_frames_inference()
    
    # Optional: Test video inference if file provided
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        results['video_inference'] = test_video_inference(video_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20} {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
