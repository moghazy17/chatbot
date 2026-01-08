"""
Base HTTP Client for API operations.

Provides shared HTTP client functionality with retry logic, timeout handling,
and error management.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .exceptions import (
    APIConnectionError,
    APIAuthenticationError,
    APIValidationError,
    APINotFoundError,
    APIServerError,
    APITimeoutError
)


class BaseAPIClient(ABC):
    """Abstract base class for API clients with automatic retry and error handling."""

    def __init__(self, base_url: str, timeout: int = 30, max_retries: int = 3):
        """
        Initialize the API client.

        Args:
            base_url: Base URL for API requests
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = self._create_session(max_retries)

    def _create_session(self, max_retries: int) -> requests.Session:
        """
        Create a requests session with retry logic.

        Args:
            max_retries: Maximum number of retry attempts

        Returns:
            Configured requests Session object
        """
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1.5,  # Exponential backoff: 0s, 1.5s, 3.75s, ...
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
            allowed_methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
        )

        # Mount retry adapter for both HTTP and HTTPS
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    @abstractmethod
    def get_headers(self) -> Dict[str, str]:
        """
        Get HTTP headers for API requests.

        Must be implemented by subclasses to provide API-specific headers.

        Returns:
            Dictionary of HTTP headers
        """
        pass

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an HTTP request with error handling.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            data: Request body data (for POST, PUT, PATCH)
            params: Query parameters

        Returns:
            Response data as dictionary

        Raises:
            APIAuthenticationError: For 401/403 status codes
            APIValidationError: For 400 status code
            APINotFoundError: For 404 status code
            APIServerError: For 500+ status codes
            APITimeoutError: When request times out
            APIConnectionError: When connection fails
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                headers=self.get_headers(),
                timeout=self.timeout
            )

            # Handle specific HTTP error codes
            if response.status_code in (401, 403):
                raise APIAuthenticationError(
                    "Authentication failed. Please check your API credentials.",
                    status_code=response.status_code
                )
            elif response.status_code == 400:
                error_message = "Invalid request data"
                try:
                    error_data = response.json()
                    error_message = error_data.get('message', error_message)
                except Exception:
                    pass
                raise APIValidationError(error_message, status_code=400)
            elif response.status_code == 404:
                raise APINotFoundError(
                    "Resource not found",
                    status_code=404
                )
            elif response.status_code >= 500:
                raise APIServerError(
                    "Server error occurred. Please try again later.",
                    status_code=response.status_code
                )

            # Raise for any other HTTP errors
            response.raise_for_status()

            # Return JSON response
            return response.json()

        except requests.Timeout:
            raise APITimeoutError("Request timed out. Please try again.")
        except requests.ConnectionError:
            raise APIConnectionError(
                "Failed to connect to the API server. Please check your network connection."
            )

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a GET request.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            Response data as dictionary
        """
        return self._request("GET", endpoint, params=params)

    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a POST request.

        Args:
            endpoint: API endpoint path
            data: Request body data

        Returns:
            Response data as dictionary
        """
        return self._request("POST", endpoint, data=data)

    def put(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a PUT request.

        Args:
            endpoint: API endpoint path
            data: Request body data

        Returns:
            Response data as dictionary
        """
        return self._request("PUT", endpoint, data=data)

    def delete(self, endpoint: str) -> Dict[str, Any]:
        """
        Make a DELETE request.

        Args:
            endpoint: API endpoint path

        Returns:
            Response data as dictionary
        """
        return self._request("DELETE", endpoint)
