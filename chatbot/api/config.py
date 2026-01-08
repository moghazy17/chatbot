"""
API Configuration Management.

Loads and validates API credentials from environment variables.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class OracleHospitalityConfig:
    """Configuration for Oracle Hospitality API."""

    base_url: str
    hotel_id: str
    app_key: str
    timeout: int = 30
    max_retries: int = 3
    _token_provider: Optional[Callable[[], str]] = field(default=None, repr=False)

    @classmethod
    def from_env(cls, token_provider: Optional[Callable[[], str]] = None) -> 'OracleHospitalityConfig':
        """
        Load configuration from environment variables.

        Args:
            token_provider: Optional callable that returns a valid access token.
                           If not provided, will use the TokenManager.

        Returns:
            OracleHospitalityConfig instance with loaded credentials

        Raises:
            ValueError: If required environment variables are missing
        """
        base_url = os.getenv('ORACLE_API_BASE_URL')
        hotel_id = os.getenv('ORACLE_API_HOTEL_ID')
        app_key = os.getenv('ORACLE_API_APP_KEY')

        # Validate required credentials
        if not all([base_url, hotel_id, app_key]):
            missing = [
                k for k, v in {
                    'ORACLE_API_BASE_URL': base_url,
                    'ORACLE_API_HOTEL_ID': hotel_id,
                    'ORACLE_API_APP_KEY': app_key,
                }.items() if not v
            ]
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                f"Please check your .env file."
            )

        # If no token provider given, use the token manager
        if token_provider is None:
            from .oracle_hospitality.token_manager import get_valid_token
            token_provider = get_valid_token

        return cls(
            base_url=base_url.rstrip('/'),
            hotel_id=hotel_id,
            app_key=app_key,
            timeout=int(os.getenv('API_TIMEOUT', '30')),
            max_retries=int(os.getenv('API_MAX_RETRIES', '3')),
            _token_provider=token_provider
        )

    def get_token(self) -> str:
        """
        Get a valid access token.

        Returns:
            Access token string
        """
        if self._token_provider:
            return self._token_provider()
        raise ValueError("No token provider configured")

    def get_headers(self) -> Dict[str, str]:
        """
        Build HTTP headers for Oracle API requests.

        Automatically fetches a valid token (refreshing if expired).

        Returns:
            Dictionary of HTTP headers with authentication and content type
        """
        return {
            'Content-Type': 'application/json',
            'x-hotelid': self.hotel_id,
            'x-app-key': self.app_key,
            'Authorization': f'Bearer {self.get_token()}'
        }
