"""
Oracle Hospitality API Client.

Provides methods for interacting with the Oracle Hospitality API
for service requests and reservation management.
"""

from typing import Dict, Any
from ..client import BaseAPIClient
from ..config import OracleHospitalityConfig
from .models import ServiceRequestInput, ServiceRequestResponse, ReservationDetails


class OracleHospitalityClient(BaseAPIClient):
    """Client for Oracle Hospitality API operations."""

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

    def get_headers(self) -> Dict[str, str]:
        """
        Get HTTP headers for Oracle API requests.

        Returns:
            Dictionary of HTTP headers with authentication
        """
        return self.config.get_headers()

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
