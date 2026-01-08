"""
Create Service Request Tool.

This tool allows the chatbot to create service requests for hotel guests.
"""

from typing import Optional
from chatbot.tools import tools_registry
from ..exceptions import (
    APIAuthenticationError,
    APIValidationError,
    APITimeoutError,
    APIConnectionError,
    APIServerError,
)
from .models import ServiceRequestInput
from ._client_init import get_client, get_api_logger


@tools_registry.register
def create_service_request(
    request_code: str,
    room_number: str,
    comment: str,
    reservation_id: Optional[str] = None,
    priority: str = "MEDIUM",
    department_code: str = "HSK"
) -> str:
    """
    Create a service request for a hotel guest.

    Use this tool when a guest needs something in their room or requests a service.
    This is the primary way to handle guest requests like towels, water, room cleaning, maintenance, etc.

    Args:
        request_code: Type of service request. Common codes:
            - 'TOWEL' for towels
            - 'WATER' for water/beverages
            - 'CLEAN' for room cleaning
            - 'MAINT' for maintenance issues
            - 'AMENITY' for amenities
        room_number: The guest's room number (e.g., '1000', '2045')
        comment: Detailed description of what the guest needs. Be specific about quantities and details.
        reservation_id: Optional reservation ID if known. Use this when you have the guest's reservation number.
        priority: Request priority level. Options:
            - 'LOW' for non-urgent requests
            - 'MEDIUM' for standard requests (default)
            - 'HIGH' for urgent requests
        department_code: Department to handle the request. Options:
            - 'HSK' for Housekeeping (default) - use for towels, cleaning, amenities
            - 'MAINT' for Maintenance - use for repairs, technical issues
            - 'FDSK' for Front Desk - use for general inquiries
            - 'FB' for Food & Beverage - use for room service
            - 'CONCIERGE' for concierge services

    Returns:
        A confirmation message with the service request details, or an error message if the request failed.

    Examples:
        Guest needs fresh towels in room 1000:
            request_code='TOWEL', room_number='1000', comment='3 fresh bath towels needed'

        Guest needs water bottles in room 2045 urgently:
            request_code='WATER', room_number='2045', comment='2 bottles of water', priority='HIGH'

        Maintenance issue in room 1234:
            request_code='MAINT', room_number='1234', comment='AC not working', priority='HIGH', department_code='MAINT'
    """
    # Get shared client and logger
    _client = get_client()
    api_logger = get_api_logger()

    # Log tool invocation
    tool_args = {
        'request_code': request_code,
        'room_number': room_number,
        'comment': comment,
        'reservation_id': reservation_id,
        'priority': priority,
        'department_code': department_code
    }

    print(f"\n🔧 TOOL INVOKED: create_service_request")
    print(f"   Room: {room_number} | Code: {request_code} | Priority: {priority}")

    if _client is None:
        error_msg = (
            "I apologize, but the hotel service request system is not currently configured. "
            "Please contact the front desk directly for assistance."
        )
        api_logger.log_tool_invocation(
            tool_name="create_service_request",
            args=tool_args,
            result=error_msg,
            success=False,
            error="API client not configured"
        )
        return error_msg

    try:
        # Create service request input model
        request_input = ServiceRequestInput(
            hotel_id=_client.config.hotel_id,
            code=request_code.upper(),
            room=room_number,
            comment=comment,
            reservation_id=reservation_id,
            priority=priority.upper(),
            department_code=department_code.upper()
        )

        api_logger.logger.info(f"Creating service request for room {room_number}")

        # Make API call
        response = _client.create_service_request(request_input)

        # Format success response
        message = "Service request created successfully!\n\n"
        if response.service_request_id:
            message += f"Request ID: {response.service_request_id}\n"
        if response.status:
            message += f"Status: {response.status}\n"
        message += f"\nYour {request_code.lower()} request for room {room_number} has been submitted. "
        message += "The appropriate department will handle it shortly."

        # Log successful tool invocation
        api_logger.log_tool_invocation(
            tool_name="create_service_request",
            args=tool_args,
            result=message,
            success=True
        )

        print(f"✓ Tool completed successfully")

        return message

    except APIAuthenticationError as e:
        api_logger.logger.error(f"Authentication error creating service request: {e}")
        error_msg = (
            "I encountered an authentication error with the hotel system. "
            "Please contact support to verify the system configuration."
        )
        api_logger.log_tool_invocation(
            tool_name="create_service_request",
            args=tool_args,
            result=error_msg,
            success=False,
            error=str(e)
        )
        return error_msg

    except APIValidationError as e:
        api_logger.logger.error(f"Validation error creating service request: {e}")
        error_msg = f"I couldn't create the service request due to invalid data: {e.message}. Please check the details and try again."
        api_logger.log_tool_invocation(
            tool_name="create_service_request",
            args=tool_args,
            result=error_msg,
            success=False,
            error=str(e)
        )
        return error_msg

    except APITimeoutError as e:
        api_logger.logger.error(f"Timeout creating service request: {e}")
        error_msg = "The request took too long to complete. Please try again in a moment, or contact the front desk directly."
        api_logger.log_tool_invocation(
            tool_name="create_service_request",
            args=tool_args,
            result=error_msg,
            success=False,
            error=str(e)
        )
        return error_msg

    except APIConnectionError as e:
        api_logger.logger.error(f"Connection error creating service request: {e}")
        error_msg = "I'm having trouble connecting to the hotel system right now. Please try again shortly, or contact the front desk directly for immediate assistance."
        api_logger.log_tool_invocation(
            tool_name="create_service_request",
            args=tool_args,
            result=error_msg,
            success=False,
            error=str(e)
        )
        return error_msg

    except APIServerError as e:
        api_logger.logger.error(f"Server error creating service request: {e}")
        error_msg = "The hotel system is experiencing issues at the moment. Please contact the front desk directly for assistance."
        api_logger.log_tool_invocation(
            tool_name="create_service_request",
            args=tool_args,
            result=error_msg,
            success=False,
            error=str(e)
        )
        return error_msg

    except Exception as e:
        api_logger.logger.error(f"Unexpected error creating service request: {e}", exc_info=True)
        error_msg = "I encountered an unexpected error while processing your request. Please contact the front desk directly for assistance."
        api_logger.log_tool_invocation(
            tool_name="create_service_request",
            args=tool_args,
            result=error_msg,
            success=False,
            error=str(e)
        )
        return error_msg
