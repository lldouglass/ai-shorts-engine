"""Tests for job endpoints (integration tests requiring Redis)."""

from fastapi.testclient import TestClient


class TestJobEndpoints:
    """Test the job API endpoints."""

    def test_trigger_smoke_test(self, test_client: TestClient) -> None:
        """Test triggering a smoke test job."""
        response = test_client.post("/api/v1/jobs/smoke")

        assert response.status_code == 202
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "queued"
        assert "message" in data

    def test_trigger_generate_video(self, test_client: TestClient) -> None:
        """Test triggering a video generation job."""
        response = test_client.post(
            "/api/v1/jobs/generate",
            json={
                "prompt": "A beautiful sunset over the ocean",
                "title": "Sunset Video",
                "duration_seconds": 30,
            },
        )

        assert response.status_code == 202
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "queued"

    def test_trigger_generate_video_validation(self, test_client: TestClient) -> None:
        """Test video generation input validation."""
        # Empty prompt should fail
        response = test_client.post(
            "/api/v1/jobs/generate",
            json={"prompt": "", "duration_seconds": 30},
        )
        assert response.status_code == 422

        # Duration too short
        response = test_client.post(
            "/api/v1/jobs/generate",
            json={"prompt": "Valid prompt", "duration_seconds": 5},
        )
        assert response.status_code == 422

        # Duration too long
        response = test_client.post(
            "/api/v1/jobs/generate",
            json={"prompt": "Valid prompt", "duration_seconds": 300},
        )
        assert response.status_code == 422

    def test_trigger_publish_video(self, test_client: TestClient) -> None:
        """Test triggering a video publish job."""
        response = test_client.post(
            "/api/v1/jobs/publish",
            json={
                "video_id": "test-video-123",
                "platform": "youtube",
                "title": "My Video",
                "description": "A test video",
            },
        )

        assert response.status_code == 202
        data = response.json()
        assert "task_id" in data

    def test_trigger_publish_invalid_platform(self, test_client: TestClient) -> None:
        """Test publishing with invalid platform."""
        response = test_client.post(
            "/api/v1/jobs/publish",
            json={
                "video_id": "test-video-123",
                "platform": "invalid_platform",
                "title": "My Video",
            },
        )

        assert response.status_code == 422

    def test_trigger_ingest_analytics(self, test_client: TestClient) -> None:
        """Test triggering analytics ingestion."""
        response = test_client.post(
            "/api/v1/jobs/ingest/analytics",
            json={
                "platform": "youtube",
                "platform_video_id": "abc123xyz",
            },
        )

        assert response.status_code == 202
        data = response.json()
        assert "task_id" in data

    def test_trigger_ingest_comments(self, test_client: TestClient) -> None:
        """Test triggering comments ingestion."""
        response = test_client.post(
            "/api/v1/jobs/ingest/comments",
            json={
                "platform": "tiktok",
                "platform_video_id": "tiktok123",
            },
        )

        assert response.status_code == 202
        data = response.json()
        assert "task_id" in data

    def test_get_job_status_unknown(self, test_client: TestClient) -> None:
        """Test getting status of unknown job."""
        response = test_client.get("/api/v1/jobs/unknown-task-id-12345")

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "unknown-task-id-12345"
        assert data["status"] == "pending"  # Unknown tasks show as pending
