"""Instagram OAuth 2.0 flow implementation via Facebook Login.

Uses Meta Graph API for Instagram Professional accounts (Business or Creator).
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

# Meta/Facebook OAuth endpoints
FACEBOOK_AUTH_URL = "https://www.facebook.com/v18.0/dialog/oauth"
FACEBOOK_TOKEN_URL = "https://graph.facebook.com/v18.0/oauth/access_token"
FACEBOOK_DEBUG_TOKEN_URL = "https://graph.facebook.com/debug_token"
GRAPH_API_URL = "https://graph.facebook.com/v18.0"

# Required scopes for Instagram Reels publishing
INSTAGRAM_SCOPES = [
    "instagram_basic",
    "instagram_content_publish",
    "instagram_manage_insights",
    "instagram_manage_comments",
    "pages_show_list",
    "pages_read_engagement",
]


class InstagramOAuthError(Exception):
    """Raised when Instagram OAuth flow fails."""

    pass


@dataclass
class InstagramOAuthCredentials:
    """OAuth credentials from successful Instagram authentication."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int | None = None  # Long-lived tokens don't expire for 60 days
    instagram_account_id: str | None = None
    instagram_username: str | None = None
    facebook_page_id: str | None = None


@dataclass
class InstagramOAuthConfig:
    """Instagram OAuth configuration."""

    app_id: str
    app_secret: str
    redirect_uri: str = "http://localhost:8085/instagram/callback"


def get_instagram_oauth_config() -> InstagramOAuthConfig:
    """Get Instagram OAuth config from environment variables."""
    app_id = os.environ.get("INSTAGRAM_APP_ID")
    app_secret = os.environ.get("INSTAGRAM_APP_SECRET")

    if not app_id or not app_secret:
        raise InstagramOAuthError(
            "INSTAGRAM_APP_ID and INSTAGRAM_APP_SECRET environment variables are required. "
            "Create a Facebook App at https://developers.facebook.com/apps and add Instagram Graph API."
        )

    redirect_uri = os.environ.get(
        "INSTAGRAM_REDIRECT_URI", "http://localhost:8085/instagram/callback"
    )

    return InstagramOAuthConfig(
        app_id=app_id,
        app_secret=app_secret,
        redirect_uri=redirect_uri,
    )


class InstagramCallbackAuth:
    """Instagram OAuth using local callback server via Facebook Login."""

    def __init__(self, config: InstagramOAuthConfig):
        self.config = config
        self.client = httpx.Client(timeout=30)
        self._auth_code: str | None = None
        self._state: str | None = None
        self._error: str | None = None

    def get_authorization_url(self) -> tuple[str, str]:
        """Generate Facebook OAuth authorization URL.

        Returns:
            Tuple of (authorization_url, state)
        """
        self._state = secrets.token_urlsafe(32)

        params = {
            "client_id": self.config.app_id,
            "redirect_uri": self.config.redirect_uri,
            "response_type": "code",
            "scope": ",".join(INSTAGRAM_SCOPES),
            "state": self._state,
        }

        url = f"{FACEBOOK_AUTH_URL}?{urlencode(params)}"
        return url, self._state

    def start_callback_server(self, port: int = 8085, timeout: int = 300) -> str:
        """Start local server to receive OAuth callback.

        Args:
            port: Port to listen on.
            timeout: Maximum time to wait for callback.

        Returns:
            Authorization code from callback.

        Raises:
            InstagramOAuthError: If callback fails or times out.
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
                        b"<html><body><h1>Instagram Authorization Failed</h1>"
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
                        b"<html><body><h1>Instagram Authorization Successful!</h1>"
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
            raise InstagramOAuthError(f"Authorization failed: {self._error}")

        if not self._auth_code:
            raise InstagramOAuthError("No authorization code received")

        return self._auth_code

    def exchange_code_for_tokens(self, code: str) -> InstagramOAuthCredentials:
        """Exchange authorization code for access token.

        Args:
            code: The authorization code from callback.

        Returns:
            Instagram OAuth credentials.
        """
        # Step 1: Exchange code for short-lived token
        response = self.client.get(
            FACEBOOK_TOKEN_URL,
            params={
                "client_id": self.config.app_id,
                "client_secret": self.config.app_secret,
                "redirect_uri": self.config.redirect_uri,
                "code": code,
            },
        )

        if response.status_code != 200:
            raise InstagramOAuthError(f"Token exchange failed: {response.text}")

        data = response.json()
        short_lived_token = data["access_token"]

        # Step 2: Exchange for long-lived token (60 day expiry)
        long_response = self.client.get(
            FACEBOOK_TOKEN_URL,
            params={
                "grant_type": "fb_exchange_token",
                "client_id": self.config.app_id,
                "client_secret": self.config.app_secret,
                "fb_exchange_token": short_lived_token,
            },
        )

        if long_response.status_code != 200:
            logger.warning(f"Failed to get long-lived token, using short-lived: {long_response.text}")
            access_token = short_lived_token
            expires_in = data.get("expires_in")
        else:
            long_data = long_response.json()
            access_token = long_data["access_token"]
            expires_in = long_data.get("expires_in", 5184000)  # Default 60 days

        # Step 3: Get Instagram Business Account
        instagram_info = self._get_instagram_account(access_token)

        return InstagramOAuthCredentials(
            access_token=access_token,
            expires_in=expires_in,
            instagram_account_id=instagram_info.get("instagram_account_id"),
            instagram_username=instagram_info.get("username"),
            facebook_page_id=instagram_info.get("page_id"),
        )

    def _get_instagram_account(self, access_token: str) -> dict[str, str]:
        """Get the Instagram Business/Creator account linked to Facebook Page.

        Args:
            access_token: Valid access token.

        Returns:
            Dict with instagram_account_id, username, and page_id.
        """
        # Get user's Facebook Pages
        pages_response = self.client.get(
            f"{GRAPH_API_URL}/me/accounts",
            params={
                "access_token": access_token,
                "fields": "id,name,instagram_business_account{id,username}",
            },
        )

        if pages_response.status_code != 200:
            logger.warning(f"Failed to get pages: {pages_response.text}")
            return {}

        pages_data = pages_response.json()
        pages = pages_data.get("data", [])

        if not pages:
            logger.warning("No Facebook Pages found. Instagram Business account requires a linked Page.")
            return {}

        # Find first page with Instagram account
        for page in pages:
            ig_account = page.get("instagram_business_account")
            if ig_account:
                return {
                    "instagram_account_id": ig_account["id"],
                    "username": ig_account.get("username"),
                    "page_id": page["id"],
                }

        logger.warning("No Instagram Business account linked to any Facebook Page.")
        return {}


def refresh_instagram_token(access_token: str) -> dict[str, Any]:
    """Refresh a long-lived Instagram/Facebook access token.

    Long-lived tokens can be refreshed if they haven't expired.
    The new token will be valid for 60 days.

    Args:
        access_token: The current long-lived access token.

    Returns:
        Dict with new access_token and expires_in.

    Raises:
        InstagramOAuthError: If refresh fails.
    """
    config = get_instagram_oauth_config()

    with httpx.Client(timeout=30) as client:
        response = client.get(
            FACEBOOK_TOKEN_URL,
            params={
                "grant_type": "fb_exchange_token",
                "client_id": config.app_id,
                "client_secret": config.app_secret,
                "fb_exchange_token": access_token,
            },
        )

    if response.status_code != 200:
        data = response.json()
        error = data.get("error", {})
        error_msg = error.get("message", response.text)

        if "expired" in error_msg.lower() or error.get("code") == 190:
            raise InstagramOAuthError(
                "Access token is expired. Please reconnect the account with "
                "'shorts-engine accounts connect instagram'."
            )

        raise InstagramOAuthError(f"Token refresh failed: {error_msg}")

    return response.json()


def run_instagram_oauth_flow(port: int = 8085) -> InstagramOAuthCredentials:
    """Run the complete Instagram OAuth authorization flow.

    Args:
        port: Port for local callback server.

    Returns:
        Instagram OAuth credentials on success.
    """
    import webbrowser

    config = get_instagram_oauth_config()
    # Update redirect URI with actual port
    config.redirect_uri = f"http://localhost:{port}/instagram/callback"

    auth = InstagramCallbackAuth(config)

    # Generate authorization URL
    auth_url, state = auth.get_authorization_url()

    print(f"\nOpening browser for Instagram authorization via Facebook Login...")
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
