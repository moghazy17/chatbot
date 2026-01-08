"""
API Configuration Management.

Loads and validates API credentials from environment variables.
"""

import os
from dataclasses import dataclass
from typing import Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class OracleHospitalityConfig:
    """Configuration for Oracle Hospitality API."""

    base_url: str
    hotel_id: str
    app_key: str
    bearer_token: str
    timeout: int = 30
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> 'OracleHospitalityConfig':
        """
        Load configuration from environment variables.

        Returns:
            OracleHospitalityConfig instance with loaded credentials

        Raises:
            ValueError: If required environment variables are missing
        """
        base_url = os.getenv('ORACLE_API_BASE_URL')
        hotel_id = os.getenv('ORACLE_API_HOTEL_ID')
        app_key = os.getenv('ORACLE_API_APP_KEY')
        bearer_token = os.getenv('ORACLE_API_BEARER_TOKEN')

        # Validate required credentials
        if not all([base_url, hotel_id, app_key, bearer_token]):
            missing = [
                k for k, v in {
                    'ORACLE_API_BASE_URL': base_url,
                    'ORACLE_API_HOTEL_ID': hotel_id,
                    'ORACLE_API_APP_KEY': app_key,
                    'ORACLE_API_BEARER_TOKEN': bearer_token
                }.items() if not v
            ]
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                f"Please check your .env file."
            )

        return cls(
            base_url=base_url.rstrip('/'),
            hotel_id=hotel_id,
            app_key=app_key,
            bearer_token=bearer_token,
            timeout=int(os.getenv('API_TIMEOUT', '30')),
            max_retries=int(os.getenv('API_MAX_RETRIES', '3'))
        )

    def get_headers(self) -> Dict[str, str]:
        """
        Build HTTP headers for Oracle API requests.

        Returns:
            Dictionary of HTTP headers with authentication and content type
        """
        return {
            'Content-Type': 'application/json',
            'x-hotelid': self.hotel_id,
            'x-app-key': self.app_key,
            'Authorization': f'Bearer {self.bearer_token}'
        }
