import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "AI EDA Platform" in response.json()["message"]

def test_api_docs():
    """Test that API documentation is accessible"""
    response = client.get("/docs")
    assert response.status_code == 200
