"""
Oracle Hospitality API Client.

Provides methods for interacting with the Oracle Hospitality API
for service requests and reservation management.
Automatically handles token refresh on authentication errors.
"""

from typing import Dict, Any, Optional
from ..client import BaseAPIClient
from ..config import OracleHospitalityConfig
from ..exceptions import APIAuthenticationError
from .models import ServiceRequestInput, ServiceRequestResponse, ReservationDetails


class OracleHospitalityClient(BaseAPIClient):
    """
    Client for Oracle Hospitality API operations.

    Features:
    - Automatic token refresh on 401 errors
    - Retry with new token if authentication fails
    """

    def __init__(self, config: OracleHospitalityConfig):
        """
        Initialize Oracle Hospitality API client.

        Args:
            config: Oracle Hospitality API configuration
        """
        super().__init__(
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries
        )
        self.config = config
        self._token_manager = None

    def _get_token_manager(self):
        """Lazy load the token manager."""
        if self._token_manager is None:
            from .token_manager import get_token_manager
            self._token_manager = get_token_manager()
        return self._token_manager

    def get_headers(self) -> Dict[str, str]:
        """
        Get HTTP headers for Oracle API requests.

        Returns:
            Dictionary of HTTP headers with authentication
        """
        return self.config.get_headers()

    def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a request with automatic token refresh on 401 errors.

        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request body data
            params: Query parameters

        Returns:
            Response data

        Raises:
            APIError: If request fails after retry
        """
        try:
            return self._request(method, endpoint, data, params)
        except APIAuthenticationError:
            # Token might be expired, invalidate and retry
            print("🔄 Token expired, refreshing...")
            token_manager = self._get_token_manager()
            token_manager.invalidate_token()

            # Retry the request (will use new token)
            return self._request(method, endpoint, data, params)

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a GET request with automatic token refresh.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            Response data as dictionary
        """
        return self._request_with_retry("GET", endpoint, params=params)

    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a POST request with automatic token refresh.

        Args:
            endpoint: API endpoint path
            data: Request body data

        Returns:
            Response data as dictionary
        """
        return self._request_with_retry("POST", endpoint, data=data)

    def put(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a PUT request with automatic token refresh.

        Args:
            endpoint: API endpoint path
            data: Request body data

        Returns:
            Response data as dictionary
        """
        return self._request_with_retry("PUT", endpoint, data=data)

    def delete(self, endpoint: str) -> Dict[str, Any]:
        """
        Make a DELETE request with automatic token refresh.

        Args:
            endpoint: API endpoint path

        Returns:
            Response data as dictionary
        """
        return self._request_with_retry("DELETE", endpoint)

    def create_service_request(self, request_input: ServiceRequestInput) -> ServiceRequestResponse:
        """
        Create a service request.

        Args:
            request_input: Service request input data

        Returns:
            ServiceRequestResponse with request details

        Raises:
            APIError: If the request fails
        """
        endpoint = f"/fof/v0/hotels/{self.config.hotel_id}/serviceRequests"
        request_data = request_input.to_api_dict()

        response = self.post(endpoint, request_data)
        return ServiceRequestResponse.from_api_response(response)

    def get_reservation_details(self, reservation_id: str) -> ReservationDetails:
        """
        Retrieve reservation details by ID.

        Args:
            reservation_id: The reservation ID to look up

        Returns:
            ReservationDetails with reservation information

        Raises:
            APIError: If the request fails
        """
        endpoint = f"/fof/v0/hotels/{self.config.hotel_id}/reservations/{reservation_id}"
        response = self.get(endpoint)
        return ReservationDetails.from_api_response(response)
