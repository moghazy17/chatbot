"""
Pydantic models for Oracle Hospitality API request/response validation.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class ServiceRequestInput(BaseModel):
    """Input model for creating service requests."""

    hotel_id: str = Field(..., description="Hotel ID")
    code: str = Field(..., description="Service request type code (e.g., TOWEL, WATER, CLEAN, MAINT)")
    status: str = Field(default="Open", description="Request status")
    priority: str = Field(default="MEDIUM", description="Request priority (LOW, MEDIUM, HIGH)")
    department_code: str = Field(..., description="Department code (HSK, MAINT, FDSK, FB, CONCIERGE)")
    room: str = Field(..., description="Room number")
    reservation_id: Optional[str] = Field(None, description="Reservation ID if known")
    comment: str = Field(..., description="Detailed service request description")
    open_date: Optional[str] = Field(None, description="Request open date (auto-generated if not provided)")

    @field_validator('priority')
    @classmethod
    def validate_priority(cls, v: str) -> str:
        """Validate priority is one of allowed values."""
        allowed = ['LOW', 'MEDIUM', 'HIGH']
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"Priority must be one of {allowed}, got '{v}'")
        return v_upper

    @field_validator('department_code')
    @classmethod
    def validate_department(cls, v: str) -> str:
        """Validate department code is one of allowed values."""
        allowed = ['HSK', 'MAINT', 'FDSK', 'FB', 'CONCIERGE']
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"Department must be one of {allowed}, got '{v}'")
        return v_upper

    def to_api_dict(self) -> dict:
        """
        Convert to Oracle API format.

        Returns:
            Dictionary formatted for Oracle Hospitality API
        """
        request_dict = {
            "hotelId": self.hotel_id,
            "code": self.code,
            "status": self.status,
            "priority": self.priority,
            "department": {
                "code": self.department_code
            },
            "room": self.room,
            "comment": self.comment
        }

        # Add reservation ID if provided
        if self.reservation_id:
            request_dict["reservationIdList"] = [{
                "type": "Reservation",
                "id": self.reservation_id
            }]

        # Add open date (use provided or generate current timestamp)
        if self.open_date:
            request_dict["openDate"] = self.open_date
        else:
            request_dict["openDate"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.0")

        return {"serviceRequestsDetails": [request_dict]}


class ServiceRequestResponse(BaseModel):
    """Response model for service request creation."""

    service_request_id: Optional[str] = Field(None, description="Service request ID")
    status: Optional[str] = Field(None, description="Request status")
    message: Optional[str] = Field(None, description="Response message")

    @classmethod
    def from_api_response(cls, response: dict) -> 'ServiceRequestResponse':
        """
        Parse Oracle API response.

        Args:
            response: Raw API response dictionary

        Returns:
            ServiceRequestResponse instance
        """
        # Extract service request details if available
        service_request_id = None
        status = None

        if 'serviceRequestsDetails' in response and response['serviceRequestsDetails']:
            details = response['serviceRequestsDetails'][0]
            service_request_id = details.get('serviceRequestId')
            status = details.get('status')

        return cls(
            service_request_id=service_request_id,
            status=status,
            message=response.get('message')
        )


class ReservationDetails(BaseModel):
    """Model for reservation details."""

    reservation_id: str = Field(..., description="Reservation ID")
    guest_name: Optional[str] = Field(None, description="Guest name")
    room_number: Optional[str] = Field(None, description="Room number")
    check_in_date: Optional[str] = Field(None, description="Check-in date")
    check_out_date: Optional[str] = Field(None, description="Check-out date")
    status: Optional[str] = Field(None, description="Reservation status")

    @classmethod
    def from_api_response(cls, response: dict) -> 'ReservationDetails':
        """
        Parse Oracle API response for reservation details.

        Args:
            response: Raw API response dictionary

        Returns:
            ReservationDetails instance
        """
        return cls(
            reservation_id=response.get('reservationId', ''),
            guest_name=response.get('guestName'),
            room_number=response.get('roomNumber'),
            check_in_date=response.get('checkInDate'),
            check_out_date=response.get('checkOutDate'),
            status=response.get('status')
        )
