"""Pytest configuration and fixtures."""

import os
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

# Set test environment before importing app modules
os.environ["DATABASE_URL"] = "postgresql://shorts:shorts@localhost:5432/shorts_test"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"
os.environ["CELERY_BROKER_URL"] = "redis://localhost:6379/1"
os.environ["CELERY_RESULT_BACKEND"] = "redis://localhost:6379/1"
os.environ["LOG_LEVEL"] = "WARNING"


@pytest.fixture(scope="session")
def test_client() -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI app."""
    from shorts_engine.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def video_gen_provider():
    """Get a stub video generation provider."""
    from shorts_engine.adapters.video_gen.stub import StubVideoGenProvider

    return StubVideoGenProvider()


@pytest.fixture
def renderer_provider():
    """Get a stub renderer provider."""
    from shorts_engine.adapters.renderer.stub import StubRendererProvider

    return StubRendererProvider()


@pytest.fixture
def publisher_adapter():
    """Get a stub publisher adapter."""
    from shorts_engine.adapters.publisher.stub import StubPublisherAdapter
    from shorts_engine.domain.enums import Platform

    return StubPublisherAdapter(platform=Platform.YOUTUBE)


@pytest.fixture
def analytics_adapter():
    """Get a stub analytics adapter."""
    from shorts_engine.adapters.analytics.stub import StubAnalyticsAdapter
    from shorts_engine.domain.enums import Platform

    return StubAnalyticsAdapter(platform=Platform.YOUTUBE)


@pytest.fixture
def comments_adapter():
    """Get a stub comments adapter."""
    from shorts_engine.adapters.comments.stub import StubCommentsAdapter
    from shorts_engine.domain.enums import Platform

    return StubCommentsAdapter(platform=Platform.YOUTUBE)
