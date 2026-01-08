"""
Shared Oracle Hospitality API client initialization.

This module initializes the API client once and shares it across all tool modules.
"""

from typing import Optional
from ..config import OracleHospitalityConfig
from ..logging_config import get_logger
from .client import OracleHospitalityClient

# Configure detailed logging
api_logger = get_logger("oracle_hospitality")

# Initialize Oracle API client (will be None if not configured)
_client: Optional[OracleHospitalityClient] = None

try:
    config = OracleHospitalityConfig.from_env()
    _client = OracleHospitalityClient(config)
    api_logger.logger.info("Oracle Hospitality API client initialized successfully")
    print("✓ Oracle Hospitality API tools loaded successfully")
except ValueError as e:
    api_logger.logger.warning(f"Oracle Hospitality API not configured: {e}")
    _client = None


def get_client() -> Optional[OracleHospitalityClient]:
    """Get the shared Oracle Hospitality API client."""
    return _client


def get_api_logger():
    """Get the shared API logger."""
    return api_logger
