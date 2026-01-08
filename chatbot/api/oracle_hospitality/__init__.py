"""
Oracle Hospitality API integration.

Provides tools for interacting with the Oracle Hospitality API,
including service request creation and reservation management.
"""

from .client import OracleHospitalityClient
from .models import ServiceRequestInput, ServiceRequestResponse, ReservationDetails

# Import tools (they auto-register with tools_registry via decorators)
from .create_service_request import create_service_request
from .get_reservation_details import get_reservation_details

__all__ = [
    # Client and models
    'OracleHospitalityClient',
    'ServiceRequestInput',
    'ServiceRequestResponse',
    'ReservationDetails',
    # Tools
    'create_service_request',
    'get_reservation_details',
]
