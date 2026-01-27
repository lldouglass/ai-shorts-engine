"""Video publishing adapters."""

from shorts_engine.adapters.publisher.base import (
    PublisherAdapter,
    PublishRequest,
    PublishResponse,
)
from shorts_engine.adapters.publisher.instagram import InstagramPublisher
from shorts_engine.adapters.publisher.instagram_oauth import (
    InstagramCallbackAuth,
    InstagramOAuthCredentials,
    InstagramOAuthError,
    get_instagram_oauth_config,
    refresh_instagram_token,
    run_instagram_oauth_flow,
)
from shorts_engine.adapters.publisher.stub import StubPublisherAdapter
from shorts_engine.adapters.publisher.tiktok import TikTokPublisher
from shorts_engine.adapters.publisher.tiktok_oauth import (
    TikTokCallbackAuth,
    TikTokOAuthCredentials,
    TikTokOAuthError,
    check_direct_post_capability,
    refresh_tiktok_token,
    revoke_tiktok_token,
    run_tiktok_oauth_flow,
)
from shorts_engine.adapters.publisher.youtube import (
    YouTubeAccountState,
    YouTubePublisher,
    YouTubeUploadResult,
    build_dry_run_payload,
)
from shorts_engine.adapters.publisher.youtube_oauth import (
    OAuthConfig,
    OAuthCredentials,
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
    # YouTube OAuth
    "OAuthCredentials",
    "OAuthConfig",
    "OAuthError",
    "get_oauth_config",
    "refresh_access_token",
    "run_device_flow",
    "run_local_callback_flow",
    # Instagram
    "InstagramPublisher",
    "InstagramOAuthError",
    "InstagramOAuthCredentials",
    "InstagramCallbackAuth",
    "run_instagram_oauth_flow",
    "refresh_instagram_token",
    "get_instagram_oauth_config",
    # TikTok
    "TikTokPublisher",
    "TikTokOAuthError",
    "TikTokOAuthCredentials",
    "TikTokCallbackAuth",
    "run_tiktok_oauth_flow",
    "refresh_tiktok_token",
    "check_direct_post_capability",
    "revoke_tiktok_token",
]
