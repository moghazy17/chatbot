"""
API module for chatbot integrations.

This module provides API clients and tools for various external services.
"""

from .exceptions import (
    APIError,
    APIConnectionError,
    APIAuthenticationError,
    APIValidationError,
    APINotFoundError,
    APIServerError,
    APITimeoutError
)
from .client import BaseAPIClient
from .config import OracleHospitalityConfig

__all__ = [
    'APIError',
    'APIConnectionError',
    'APIAuthenticationError',
    'APIValidationError',
    'APINotFoundError',
    'APIServerError',
    'APITimeoutError',
    'BaseAPIClient',
    'OracleHospitalityConfig'
]
