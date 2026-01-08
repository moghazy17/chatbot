"""
Custom exceptions for API operations.

This module defines a hierarchy of exceptions for handling various API errors
in a structured and user-friendly way.
"""

from typing import Optional


class APIError(Exception):
    """Base exception for all API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        """
        Initialize API error.

        Args:
            message: Human-readable error message
            status_code: HTTP status code if applicable
        """
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class APIConnectionError(APIError):
    """Raised when unable to connect to the API server."""
    pass


class APIAuthenticationError(APIError):
    """Raised when authentication fails (401, 403 status codes)."""
    pass


class APIValidationError(APIError):
    """Raised when request validation fails (400 status code)."""
    pass


class APINotFoundError(APIError):
    """Raised when a requested resource is not found (404 status code)."""
    pass


class APIServerError(APIError):
    """Raised when the server encounters an error (500+ status codes)."""
    pass


class APITimeoutError(APIError):
    """Raised when a request times out."""
    pass
