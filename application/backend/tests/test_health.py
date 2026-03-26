"""
Test Health Endpoints
"""

import pytest


def test_health_check(client):
    """Test basic health check endpoint"""
    response = client.get("/api/v1/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert data["status"] == "healthy"


def test_system_info(client):
    """Test system information endpoint"""
    response = client.get("/api/v1/health/system")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check for expected fields
    assert "python_version" in data or "status" in data


def test_model_status(client):
    """Test model status endpoint"""
    response = client.get("/api/v1/health/models")
    
    assert response.status_code == 200
    data = response.json()
    
    # Should contain status information
    assert isinstance(data, dict)
    assert "inference_service" in data or "modules" in data


def test_health_response_format(client):
    """Test health response has correct format"""
    response = client.get("/api/v1/health")
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    
    data = response.json()
    assert isinstance(data, dict)


def test_health_multiple_requests(client):
    """Test multiple health check requests"""
    for _ in range(5):
        response = client.get("/api/v1/health")
        assert response.status_code == 200


def test_invalid_health_endpoint(client):
    """Test invalid health endpoint returns 404"""
    response = client.get("/api/v1/health/invalid")
    assert response.status_code == 404
