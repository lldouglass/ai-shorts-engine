"""YouTube OAuth 2.0 flow implementation.

Supports both device flow (preferred for CLI) and local callback server.
"""

import json
import logging
import os
import secrets
import time
import webbrowser
from dataclasses import dataclass
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

logger = logging.getLogger(__name__)

# YouTube OAuth endpoints
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_DEVICE_AUTH_URL = "https://oauth2.googleapis.com/device/code"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
YOUTUBE_CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"

# Required scopes for YouTube upload
YOUTUBE_UPLOAD_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.readonly",
]


class OAuthError(Exception):
    """Raised when OAuth flow fails."""

    pass


@dataclass
class OAuthCredentials:
    """OAuth credentials from successful authentication."""

    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str
    scope: str
    channel_id: str | None = None
    channel_title: str | None = None


@dataclass
class OAuthConfig:
    """OAuth configuration."""

    client_id: str
    client_secret: str
    redirect_uri: str = "http://localhost:8085/callback"


def get_oauth_config() -> OAuthConfig:
    """Get OAuth config from environment variables."""
    client_id = os.environ.get("YOUTUBE_CLIENT_ID")
    client_secret = os.environ.get("YOUTUBE_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise OAuthError(
            "YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET environment variables are required. "
            "Create OAuth credentials at https://console.cloud.google.com/apis/credentials"
        )

    redirect_uri = os.environ.get("YOUTUBE_REDIRECT_URI", "http://localhost:8085/callback")

    return OAuthConfig(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
    )


class DeviceFlowAuth:
    """YouTube OAuth using device flow (no browser redirect needed)."""

    def __init__(self, config: OAuthConfig):
        self.config = config
        self.client = httpx.Client(timeout=30)

    def start_device_flow(self) -> dict[str, Any]:
        """Start the device authorization flow.

        Returns:
            Device code response containing user_code, verification_url, etc.
        """
        response = self.client.post(
            GOOGLE_DEVICE_AUTH_URL,
            data={
                "client_id": self.config.client_id,
                "scope": " ".join(YOUTUBE_UPLOAD_SCOPES),
            },
        )

        if response.status_code != 200:
            raise OAuthError(f"Device flow initiation failed: {response.text}")

        return response.json()

    def poll_for_token(
        self,
        device_code: str,
        interval: int = 5,
        timeout: int = 300,
    ) -> OAuthCredentials:
        """Poll for token after user completes authorization.

        Args:
            device_code: The device_code from start_device_flow.
            interval: Polling interval in seconds.
            timeout: Maximum time to wait in seconds.

        Returns:
            OAuth credentials on success.

        Raises:
            OAuthError: If authorization fails or times out.
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            response = self.client.post(
                GOOGLE_TOKEN_URL,
                data={
                    "client_id": self.config.client_id,
                    "client_secret": self.config.client_secret,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
            )

            data = response.json()

            if response.status_code == 200:
                # Success - get channel info
                channel_info = self._get_channel_info(data["access_token"])

                return OAuthCredentials(
                    access_token=data["access_token"],
                    refresh_token=data["refresh_token"],
                    expires_in=data["expires_in"],
                    token_type=data["token_type"],
                    scope=data.get("scope", ""),
                    channel_id=channel_info.get("channel_id"),
                    channel_title=channel_info.get("channel_title"),
                )

            error = data.get("error")

            if error == "authorization_pending":
                # User hasn't authorized yet, keep polling
                time.sleep(interval)
                continue
            elif error == "slow_down":
                # Increase polling interval
                interval += 5
                time.sleep(interval)
                continue
            elif error == "access_denied":
                raise OAuthError("User denied access")
            elif error == "expired_token":
                raise OAuthError("Device code expired. Please restart the authorization.")
            else:
                raise OAuthError(f"Authorization failed: {data}")

        raise OAuthError("Authorization timed out")

    def _get_channel_info(self, access_token: str) -> dict[str, str]:
        """Get the user's YouTube channel information."""
        response = self.client.get(
            YOUTUBE_CHANNELS_URL,
            params={"part": "snippet", "mine": "true"},
            headers={"Authorization": f"Bearer {access_token}"},
        )

        if response.status_code != 200:
            logger.warning(f"Failed to get channel info: {response.text}")
            return {}

        data = response.json()
        items = data.get("items", [])

        if not items:
            return {}

        channel = items[0]
        return {
            "channel_id": channel["id"],
            "channel_title": channel["snippet"]["title"],
        }


class LocalCallbackAuth:
    """YouTube OAuth using local callback server."""

    def __init__(self, config: OAuthConfig):
        self.config = config
        self.client = httpx.Client(timeout=30)
        self._auth_code: str | None = None
        self._state: str | None = None
        self._error: str | None = None

    def get_authorization_url(self) -> tuple[str, str]:
        """Generate authorization URL.

        Returns:
            Tuple of (authorization_url, state)
        """
        self._state = secrets.token_urlsafe(32)

        params = {
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "response_type": "code",
            "scope": " ".join(YOUTUBE_UPLOAD_SCOPES),
            "access_type": "offline",
            "prompt": "consent",  # Force refresh token
            "state": self._state,
        }

        url = f"{GOOGLE_AUTH_URL}?{urlencode(params)}"
        return url, self._state

    def start_callback_server(self, port: int = 8085, timeout: int = 300) -> str:
        """Start local server to receive OAuth callback.

        Args:
            port: Port to listen on.
            timeout: Maximum time to wait for callback.

        Returns:
            Authorization code from callback.

        Raises:
            OAuthError: If callback fails or times out.
        """
        outer = self

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)

                if "error" in params:
                    outer._error = params["error"][0]
                    self.send_response(400)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(
                        b"<html><body><h1>Authorization Failed</h1>"
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
                        b"<html><body><h1>Authorization Successful!</h1>"
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
            raise OAuthError(f"Authorization failed: {self._error}")

        if not self._auth_code:
            raise OAuthError("No authorization code received")

        return self._auth_code

    def exchange_code_for_tokens(self, code: str) -> OAuthCredentials:
        """Exchange authorization code for tokens.

        Args:
            code: The authorization code from callback.

        Returns:
            OAuth credentials.
        """
        response = self.client.post(
            GOOGLE_TOKEN_URL,
            data={
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": self.config.redirect_uri,
            },
        )

        if response.status_code != 200:
            raise OAuthError(f"Token exchange failed: {response.text}")

        data = response.json()

        if "refresh_token" not in data:
            raise OAuthError(
                "No refresh token received. This may happen if the app was already authorized. "
                "Revoke access at https://myaccount.google.com/permissions and try again."
            )

        # Get channel info
        channel_info = self._get_channel_info(data["access_token"])

        return OAuthCredentials(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_in=data["expires_in"],
            token_type=data["token_type"],
            scope=data.get("scope", ""),
            channel_id=channel_info.get("channel_id"),
            channel_title=channel_info.get("channel_title"),
        )

    def _get_channel_info(self, access_token: str) -> dict[str, str]:
        """Get the user's YouTube channel information."""
        response = self.client.get(
            YOUTUBE_CHANNELS_URL,
            params={"part": "snippet", "mine": "true"},
            headers={"Authorization": f"Bearer {access_token}"},
        )

        if response.status_code != 200:
            logger.warning(f"Failed to get channel info: {response.text}")
            return {}

        data = response.json()
        items = data.get("items", [])

        if not items:
            return {}

        channel = items[0]
        return {
            "channel_id": channel["id"],
            "channel_title": channel["snippet"]["title"],
        }


def refresh_access_token(refresh_token: str) -> dict[str, Any]:
    """Refresh an access token using a refresh token.

    Args:
        refresh_token: The refresh token.

    Returns:
        Dict with new access_token and expires_in.

    Raises:
        OAuthError: If refresh fails.
    """
    config = get_oauth_config()

    with httpx.Client(timeout=30) as client:
        response = client.post(
            GOOGLE_TOKEN_URL,
            data={
                "client_id": config.client_id,
                "client_secret": config.client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
        )

    if response.status_code != 200:
        data = response.json()
        error = data.get("error", "unknown")
        error_desc = data.get("error_description", response.text)

        if error == "invalid_grant":
            raise OAuthError(
                "Refresh token is invalid or expired. "
                "Please reconnect the account with 'shorts-engine accounts connect youtube'."
            )

        raise OAuthError(f"Token refresh failed: {error_desc}")

    return response.json()


def run_device_flow() -> OAuthCredentials:
    """Run the complete device flow authorization.

    Returns:
        OAuth credentials on success.
    """
    config = get_oauth_config()
    auth = DeviceFlowAuth(config)

    # Start device flow
    device_data = auth.start_device_flow()

    user_code = device_data["user_code"]
    verification_url = device_data["verification_url"]
    device_code = device_data["device_code"]
    interval = device_data.get("interval", 5)
    expires_in = device_data.get("expires_in", 1800)

    print(f"\n1. Go to: {verification_url}")
    print(f"2. Enter code: {user_code}")
    print(f"\nWaiting for authorization (expires in {expires_in // 60} minutes)...")

    # Try to open browser
    try:
        webbrowser.open(verification_url)
    except Exception:
        pass  # Browser open is optional

    # Poll for token
    return auth.poll_for_token(device_code, interval=interval, timeout=expires_in)


def run_local_callback_flow(port: int = 8085) -> OAuthCredentials:
    """Run the complete local callback authorization flow.

    Args:
        port: Port for local callback server.

    Returns:
        OAuth credentials on success.
    """
    config = get_oauth_config()
    config.redirect_uri = f"http://localhost:{port}/callback"

    auth = LocalCallbackAuth(config)

    # Generate authorization URL
    auth_url, state = auth.get_authorization_url()

    print(f"\nOpening browser for authorization...")
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
    return auth.exchange_code_for_tokens(code)
