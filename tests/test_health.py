"""Tests for health endpoints."""

from fastapi.testclient import TestClient


def test_health_endpoint(test_client: TestClient) -> None:
    """Test the basic health endpoint."""
    response = test_client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_liveness_endpoint(test_client: TestClient) -> None:
    """Test the liveness probe endpoint."""
    response = test_client.get("/health/live")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"


def test_root_endpoint(test_client: TestClient) -> None:
    """Test the root endpoint."""
    response = test_client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "docs" in data
