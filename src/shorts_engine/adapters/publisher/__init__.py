"""Video publishing adapters."""

from shorts_engine.adapters.publisher.base import (
    PublisherAdapter,
    PublishRequest,
    PublishResponse,
)
from shorts_engine.adapters.publisher.stub import StubPublisherAdapter
from shorts_engine.adapters.publisher.youtube import (
    YouTubePublisher,
    YouTubeAccountState,
    YouTubeUploadResult,
    build_dry_run_payload,
)
from shorts_engine.adapters.publisher.youtube_oauth import (
    OAuthCredentials,
    OAuthConfig,
    OAuthError,
    get_oauth_config,
    refresh_access_token,
    run_device_flow,
    run_local_callback_flow,
)

__all__ = [
    # Base
    "PublisherAdapter",
    "PublishRequest",
    "PublishResponse",
    # Stub
    "StubPublisherAdapter",
    # YouTube
    "YouTubePublisher",
    "YouTubeAccountState",
    "YouTubeUploadResult",
    "build_dry_run_payload",
    # OAuth
    "OAuthCredentials",
    "OAuthConfig",
    "OAuthError",
    "get_oauth_config",
    "refresh_access_token",
    "run_device_flow",
    "run_local_callback_flow",
]
