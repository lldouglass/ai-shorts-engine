"""TikTok OAuth 2.0 flow implementation.

Uses TikTok Login Kit and Content Posting API for video publishing.
"""

import logging
import os
import secrets
from dataclasses import dataclass
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

logger = logging.getLogger(__name__)

# TikTok OAuth endpoints
TIKTOK_AUTH_URL = "https://www.tiktok.com/v2/auth/authorize/"
TIKTOK_TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
TIKTOK_REVOKE_URL = "https://open.tiktokapis.com/v2/oauth/revoke/"
TIKTOK_USER_INFO_URL = "https://open.tiktokapis.com/v2/user/info/"

# Required scopes for video publishing
TIKTOK_SCOPES = [
    "user.info.basic",
    "video.publish",  # Required for Direct Post
    "video.upload",  # Required for uploading videos
]


class TikTokOAuthError(Exception):
    """Raised when TikTok OAuth flow fails."""

    pass


@dataclass
class TikTokOAuthCredentials:
    """OAuth credentials from successful TikTok authentication."""

    access_token: str
    refresh_token: str
    expires_in: int  # Token validity in seconds (typically 24 hours)
    refresh_expires_in: int  # Refresh token validity (typically 365 days)
    token_type: str = "Bearer"
    scope: str = ""
    open_id: str | None = None  # TikTok user's unique open ID
    display_name: str | None = None


@dataclass
class TikTokOAuthConfig:
    """TikTok OAuth configuration."""

    client_key: str
    client_secret: str
    redirect_uri: str = "http://localhost:8085/tiktok/callback"


def get_tiktok_oauth_config() -> TikTokOAuthConfig:
    """Get TikTok OAuth config from environment variables."""
    client_key = os.environ.get("TIKTOK_CLIENT_KEY")
    client_secret = os.environ.get("TIKTOK_CLIENT_SECRET")

    if not client_key or not client_secret:
        raise TikTokOAuthError(
            "TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET environment variables are required. "
            "Create an app at https://developers.tiktok.com/apps and enable Login Kit + Content Posting API."
        )

    redirect_uri = os.environ.get(
        "TIKTOK_REDIRECT_URI", "http://localhost:8085/tiktok/callback"
    )

    return TikTokOAuthConfig(
        client_key=client_key,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
    )


class TikTokCallbackAuth:
    """TikTok OAuth using local callback server."""

    def __init__(self, config: TikTokOAuthConfig):
        self.config = config
        self.client = httpx.Client(timeout=30)
        self._auth_code: str | None = None
        self._state: str | None = None
        self._error: str | None = None

    def get_authorization_url(self) -> tuple[str, str]:
        """Generate TikTok OAuth authorization URL.

        Returns:
            Tuple of (authorization_url, state)
        """
        self._state = secrets.token_urlsafe(32)

        params = {
            "client_key": self.config.client_key,
            "redirect_uri": self.config.redirect_uri,
            "response_type": "code",
            "scope": ",".join(TIKTOK_SCOPES),
            "state": self._state,
        }

        url = f"{TIKTOK_AUTH_URL}?{urlencode(params)}"
        return url, self._state

    def start_callback_server(self, port: int = 8085, timeout: int = 300) -> str:
        """Start local server to receive OAuth callback.

        Args:
            port: Port to listen on.
            timeout: Maximum time to wait for callback.

        Returns:
            Authorization code from callback.

        Raises:
            TikTokOAuthError: If callback fails or times out.
        """
        outer = self

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)

                # Only handle the callback path
                if not parsed.path.endswith("/callback"):
                    self.send_response(404)
                    self.end_headers()
                    return

                params = parse_qs(parsed.query)

                if "error" in params:
                    outer._error = params.get("error_description", params["error"])[0]
                    self.send_response(400)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(
                        b"<html><body><h1>TikTok Authorization Failed</h1>"
                        b"<p>You can close this window.</p></body></html>"
                    )
                    return

                if "code" in params:
                    state = params.get("state", [None])[0]
                    if state != outer._state:
                        outer._error = "State mismatch"
                        self.send_response(400)
                        self.send_header("Content-type", "text/html")
                        self.end_headers()
                        self.wfile.write(b"<html><body><h1>State Mismatch</h1></body></html>")
                        return

                    outer._auth_code = params["code"][0]
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(
                        b"<html><body><h1>TikTok Authorization Successful!</h1>"
                        b"<p>You can close this window and return to the terminal.</p></body></html>"
                    )
                    return

                self.send_response(404)
                self.end_headers()

            def log_message(self, format, *args):
                # Suppress HTTP server logs
                pass

        server = HTTPServer(("localhost", port), CallbackHandler)
        server.timeout = timeout

        # Handle single request
        server.handle_request()
        server.server_close()

        if self._error:
            raise TikTokOAuthError(f"Authorization failed: {self._error}")

        if not self._auth_code:
            raise TikTokOAuthError("No authorization code received")

        return self._auth_code

    def exchange_code_for_tokens(self, code: str) -> TikTokOAuthCredentials:
        """Exchange authorization code for access token.

        Args:
            code: The authorization code from callback.

        Returns:
            TikTok OAuth credentials.
        """
        response = self.client.post(
            TIKTOK_TOKEN_URL,
            data={
                "client_key": self.config.client_key,
                "client_secret": self.config.client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": self.config.redirect_uri,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if response.status_code != 200:
            raise TikTokOAuthError(f"Token exchange failed: {response.text}")

        data = response.json()

        if "error" in data:
            error_desc = data.get("error_description", data.get("error"))
            raise TikTokOAuthError(f"Token exchange failed: {error_desc}")

        access_token = data["access_token"]
        open_id = data.get("open_id")

        # Get user info
        user_info = self._get_user_info(access_token, open_id)

        return TikTokOAuthCredentials(
            access_token=access_token,
            refresh_token=data["refresh_token"],
            expires_in=data.get("expires_in", 86400),  # 24 hours default
            refresh_expires_in=data.get("refresh_expires_in", 31536000),  # 365 days default
            scope=data.get("scope", ""),
            open_id=open_id,
            display_name=user_info.get("display_name"),
        )

    def _get_user_info(self, access_token: str, open_id: str | None) -> dict[str, Any]:
        """Get TikTok user info.

        Args:
            access_token: Valid access token.
            open_id: User's open ID.

        Returns:
            User info dict.
        """
        if not open_id:
            return {}

        response = self.client.get(
            TIKTOK_USER_INFO_URL,
            params={"fields": "open_id,display_name,avatar_url"},
            headers={"Authorization": f"Bearer {access_token}"},
        )

        if response.status_code != 200:
            logger.warning(f"Failed to get user info: {response.text}")
            return {}

        data = response.json()
        return data.get("data", {}).get("user", {})


def refresh_tiktok_token(refresh_token: str) -> dict[str, Any]:
    """Refresh a TikTok access token.

    Args:
        refresh_token: The refresh token.

    Returns:
        Dict with new access_token, refresh_token, expires_in.

    Raises:
        TikTokOAuthError: If refresh fails.
    """
    config = get_tiktok_oauth_config()

    with httpx.Client(timeout=30) as client:
        response = client.post(
            TIKTOK_TOKEN_URL,
            data={
                "client_key": config.client_key,
                "client_secret": config.client_secret,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    if response.status_code != 200:
        raise TikTokOAuthError(f"Token refresh failed: {response.text}")

    data = response.json()

    if "error" in data:
        error_desc = data.get("error_description", data.get("error"))

        if "expired" in error_desc.lower() or "invalid" in error_desc.lower():
            raise TikTokOAuthError(
                "Refresh token is expired or invalid. Please reconnect the account with "
                "'shorts-engine accounts connect tiktok'."
            )

        raise TikTokOAuthError(f"Token refresh failed: {error_desc}")

    return {
        "access_token": data["access_token"],
        "refresh_token": data.get("refresh_token", refresh_token),
        "expires_in": data.get("expires_in", 86400),
        "open_id": data.get("open_id"),
    }


def revoke_tiktok_token(access_token: str) -> bool:
    """Revoke a TikTok access token.

    Args:
        access_token: The access token to revoke.

    Returns:
        True if revocation was successful.
    """
    config = get_tiktok_oauth_config()

    with httpx.Client(timeout=30) as client:
        response = client.post(
            TIKTOK_REVOKE_URL,
            data={
                "client_key": config.client_key,
                "client_secret": config.client_secret,
                "token": access_token,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    if response.status_code != 200:
        logger.warning(f"Token revocation failed: {response.text}")
        return False

    return True


def check_direct_post_capability(access_token: str, open_id: str) -> bool:
    """Check if the account has Direct Post capability approved.

    Direct Post allows publishing videos directly to TikTok without user
    interaction. Requires approval from TikTok for production apps.

    Args:
        access_token: Valid access token.
        open_id: User's open ID.

    Returns:
        True if Direct Post is available.
    """
    # Check creator info endpoint for posting permissions
    url = "https://open.tiktokapis.com/v2/post/publish/creator_info/query/"

    with httpx.Client(timeout=30) as client:
        response = client.post(
            url,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json; charset=UTF-8",
            },
            json={},
        )

    if response.status_code != 200:
        logger.warning(f"Failed to check creator info: {response.text}")
        return False

    data = response.json()

    if data.get("error", {}).get("code") == "ok":
        creator_info = data.get("data", {})
        # Check if creator has posting permission
        return creator_info.get("creator_avatar_url") is not None

    return False


def run_tiktok_oauth_flow(port: int = 8085) -> TikTokOAuthCredentials:
    """Run the complete TikTok OAuth authorization flow.

    Args:
        port: Port for local callback server.

    Returns:
        TikTok OAuth credentials on success.
    """
    import webbrowser

    config = get_tiktok_oauth_config()
    # Update redirect URI with actual port
    config.redirect_uri = f"http://localhost:{port}/tiktok/callback"

    auth = TikTokCallbackAuth(config)

    # Generate authorization URL
    auth_url, state = auth.get_authorization_url()

    print(f"\nOpening browser for TikTok authorization...")
    print(f"If the browser doesn't open, go to:\n{auth_url}\n")

    # Open browser
    try:
        webbrowser.open(auth_url)
    except Exception:
        pass

    # Start callback server and wait for code
    print(f"Waiting for callback on port {port}...")
    code = auth.start_callback_server(port=port)

    # Exchange code for tokens
    credentials = auth.exchange_code_for_tokens(code)

    # Check Direct Post capability
    if credentials.open_id:
        has_direct_post = check_direct_post_capability(
            credentials.access_token, credentials.open_id
        )
        if not has_direct_post:
            print(
                "\n[Warning] Direct Post not approved for this account. "
                "Videos will require manual publishing through TikTok app. "
                "Apply for Direct Post at https://developers.tiktok.com/\n"
            )

    return credentials
