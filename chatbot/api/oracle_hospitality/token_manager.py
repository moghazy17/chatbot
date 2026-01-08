"""
Oracle Hospitality OAuth Token Manager.

Handles automatic token generation and refresh for the Oracle Hospitality API.
Only refreshes the token when it's expired or about to expire.
"""

import os
import time
import base64
import json
import requests
from datetime import datetime, timedelta
from typing import Optional, Tuple
from dataclasses import dataclass
from threading import Lock
from dotenv import load_dotenv

from ..logging_config import get_logger

load_dotenv()

# Get logger
logger = get_logger("oracle_token_manager")


@dataclass
class TokenInfo:
    """Holds token information."""
    access_token: str
    expires_at: datetime
    token_type: str = "Bearer"


class OracleTokenManager:
    """
    Manages OAuth tokens for Oracle Hospitality API.

    Features:
    - Automatic token refresh when expired
    - Thread-safe token access
    - Token expiration buffer (refreshes 5 minutes before expiry)
    - Caches token to minimize API calls
    """

    # Buffer time before expiration to refresh (in seconds)
    EXPIRATION_BUFFER = 300  # 5 minutes

    def __init__(
        self,
        oauth_url: Optional[str] = None,
        app_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ):
        """
        Initialize the token manager.

        Args:
            oauth_url: OAuth token endpoint URL
            app_key: Application key for x-app-key header
            username: OAuth username
            password: OAuth password
            client_id: Client ID for Basic auth (optional)
            client_secret: Client secret for Basic auth (optional)
        """
        self.oauth_url = oauth_url or os.getenv(
            'ORACLE_OAUTH_URL',
            'https://mtecn1uat.hospitality-api.eu-frankfurt-1.ocs.oc-test.com/oauth/v1/tokens'
        )
        self.app_key = app_key or os.getenv('ORACLE_API_APP_KEY')
        self.username = username or os.getenv('ORACLE_OAUTH_USERNAME')
        self.password = password or os.getenv('ORACLE_OAUTH_PASSWORD')
        self.client_id = client_id or os.getenv('ORACLE_CLIENT_ID', '')
        self.client_secret = client_secret or os.getenv('ORACLE_CLIENT_SECRET', '')

        self._token_info: Optional[TokenInfo] = None
        self._lock = Lock()

        # Validate required credentials
        self._validate_credentials()

    def _validate_credentials(self):
        """Validate that required credentials are present."""
        missing = []
        if not self.oauth_url:
            missing.append('ORACLE_OAUTH_URL')
        if not self.app_key:
            missing.append('ORACLE_API_APP_KEY')
        if not self.username:
            missing.append('ORACLE_OAUTH_USERNAME')
        if not self.password:
            missing.append('ORACLE_OAUTH_PASSWORD')

        if missing:
            raise ValueError(
                f"Missing required OAuth credentials: {', '.join(missing)}. "
                f"Please set them in your .env file."
            )

    def _get_basic_auth_header(self) -> str:
        """
        Generate Basic Authorization header.

        Returns:
            Basic auth header value
        """
        if self.client_id and self.client_secret:
            credentials = f"{self.client_id}:{self.client_secret}"
        else:
            # Use username:password as fallback
            credentials = f"{self.username}:{self.password}"

        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"

    def _parse_jwt_expiration(self, token: str) -> datetime:
        """
        Parse JWT token to extract expiration time.

        Args:
            token: JWT access token

        Returns:
            Expiration datetime
        """
        try:
            # JWT format: header.payload.signature
            parts = token.split('.')
            if len(parts) != 3:
                # Not a JWT, assume 1 hour expiration
                return datetime.now() + timedelta(hours=1)

            # Decode payload (add padding if needed)
            payload = parts[1]
            padding = 4 - len(payload) % 4
            if padding != 4:
                payload += '=' * padding

            decoded = base64.urlsafe_b64decode(payload)
            payload_data = json.loads(decoded)

            # Get expiration timestamp
            exp = payload_data.get('exp')
            if exp:
                return datetime.fromtimestamp(exp)

            # Fallback to 1 hour if no exp claim
            return datetime.now() + timedelta(hours=1)

        except Exception as e:
            logger.logger.warning(f"Failed to parse JWT expiration: {e}")
            # Default to 1 hour expiration
            return datetime.now() + timedelta(hours=1)

    def _fetch_new_token(self) -> TokenInfo:
        """
        Fetch a new access token from the OAuth endpoint.

        Returns:
            TokenInfo with new access token

        Raises:
            Exception: If token fetch fails
        """
        logger.logger.info("Fetching new OAuth token...")
        print("🔑 Fetching new OAuth token...")

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'x-app-key': self.app_key,
            'Authorization': self._get_basic_auth_header(),
        }

        data = {
            'username': self.username,
            'password': self.password,
            'grant_type': 'password',
        }

        try:
            response = requests.post(
                self.oauth_url,
                headers=headers,
                data=data,
                timeout=30
            )

            if response.status_code != 200:
                error_msg = f"OAuth token request failed with status {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f": {error_data}"
                except:
                    error_msg += f": {response.text}"
                logger.logger.error(error_msg)
                raise Exception(error_msg)

            token_data = response.json()
            access_token = token_data.get('access_token')

            if not access_token:
                raise Exception("No access_token in OAuth response")

            # Parse expiration from JWT or use expires_in
            expires_in = token_data.get('expires_in')
            if expires_in:
                expires_at = datetime.now() + timedelta(seconds=int(expires_in))
            else:
                expires_at = self._parse_jwt_expiration(access_token)

            token_info = TokenInfo(
                access_token=access_token,
                expires_at=expires_at,
                token_type=token_data.get('token_type', 'Bearer')
            )

            logger.logger.info(f"OAuth token obtained, expires at: {expires_at}")
            print(f"✓ OAuth token obtained (expires: {expires_at.strftime('%Y-%m-%d %H:%M:%S')})")

            return token_info

        except requests.RequestException as e:
            logger.logger.error(f"OAuth request failed: {e}")
            raise Exception(f"Failed to fetch OAuth token: {e}")

    def _is_token_expired(self) -> bool:
        """
        Check if the current token is expired or about to expire.

        Returns:
            True if token needs refresh, False otherwise
        """
        if self._token_info is None:
            return True

        # Check if token expires within the buffer time
        buffer_time = timedelta(seconds=self.EXPIRATION_BUFFER)
        return datetime.now() >= (self._token_info.expires_at - buffer_time)

    def get_token(self) -> str:
        """
        Get a valid access token, refreshing if necessary.

        This method is thread-safe.

        Returns:
            Valid access token string
        """
        with self._lock:
            if self._is_token_expired():
                logger.logger.debug("Token expired or missing, fetching new token")
                self._token_info = self._fetch_new_token()
            else:
                logger.logger.debug("Using cached token")

            return self._token_info.access_token

    def get_authorization_header(self) -> str:
        """
        Get the Authorization header value with a valid token.

        Returns:
            Bearer token authorization header value
        """
        token = self.get_token()
        return f"Bearer {token}"

    def invalidate_token(self):
        """
        Invalidate the current token, forcing a refresh on next request.

        Useful when an API call returns 401 Unauthorized.
        """
        with self._lock:
            self._token_info = None
            logger.logger.info("Token invalidated, will refresh on next request")

    def get_token_info(self) -> Optional[Tuple[str, datetime]]:
        """
        Get current token info without refreshing.

        Returns:
            Tuple of (token, expires_at) or None if no token
        """
        if self._token_info:
            return (self._token_info.access_token, self._token_info.expires_at)
        return None


# Global token manager instance (lazy initialization)
_token_manager: Optional[OracleTokenManager] = None


def get_token_manager() -> OracleTokenManager:
    """
    Get or create the global token manager instance.

    Returns:
        OracleTokenManager instance
    """
    global _token_manager
    if _token_manager is None:
        _token_manager = OracleTokenManager()
    return _token_manager


def get_valid_token() -> str:
    """
    Convenience function to get a valid token.

    Returns:
        Valid access token string
    """
    return get_token_manager().get_token()
