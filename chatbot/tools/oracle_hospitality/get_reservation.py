"""
Get Reservation Details Tool.

This tool allows the chatbot to retrieve reservation details for hotel guests.
"""

from ..registry import tools_registry
from ...api.exceptions import (
    APIAuthenticationError,
    APITimeoutError,
    APIConnectionError,
    APIServerError,
    APINotFoundError,
)
from ...api.oracle_hospitality._client_init import get_client, get_api_logger


@tools_registry.register
def get_reservation_details(reservation_id: str) -> str:
    """
    Retrieve details about a hotel reservation.

    Use this tool to look up information about a guest's reservation when they provide
    their reservation ID or confirmation number.

    Args:
        reservation_id: The reservation ID or confirmation number (e.g., '176478')

    Returns:
        Reservation details including guest name, room number, check-in/out dates,
        and reservation status. Returns an error message if the reservation is not found.

    Examples:
        Look up reservation 176478:
            reservation_id='176478'

        Check details for confirmation number ABC123:
            reservation_id='ABC123'
    """
    # Get shared client and logger
    _client = get_client()
    api_logger = get_api_logger()

    # Log tool invocation
    tool_args = {'reservation_id': reservation_id}

    print(f"\n🔧 TOOL INVOKED: get_reservation_details")
    print(f"   Reservation ID: {reservation_id}")

    if _client is None:
        error_msg = (
            "I apologize, but the hotel reservation system is not currently configured. "
            "Please contact the front desk directly for reservation information."
        )
        api_logger.log_tool_invocation(
            tool_name="get_reservation_details",
            args=tool_args,
            result=error_msg,
            success=False,
            error="API client not configured"
        )
        return error_msg

    try:
        api_logger.logger.info(f"Retrieving reservation details for ID: {reservation_id}")

        # Make API call
        reservation = _client.get_reservation_details(reservation_id)

        # Format response
        message = f"Reservation Details for ID: {reservation.reservation_id}\n\n"

        if reservation.guest_name:
            message += f"Guest: {reservation.guest_name}\n"
        if reservation.room_number:
            message += f"Room: {reservation.room_number}\n"
        if reservation.check_in_date:
            message += f"Check-in: {reservation.check_in_date}\n"
        if reservation.check_out_date:
            message += f"Check-out: {reservation.check_out_date}\n"
        if reservation.status:
            message += f"Status: {reservation.status}\n"

        # Log successful tool invocation
        api_logger.log_tool_invocation(
            tool_name="get_reservation_details",
            args=tool_args,
            result=message,
            success=True
        )

        print(f"✓ Tool completed successfully")

        return message

    except APINotFoundError as e:
        api_logger.logger.warning(f"Reservation not found: {reservation_id}")
        error_msg = f"I couldn't find a reservation with ID '{reservation_id}'. Please check the reservation number and try again, or contact the front desk for assistance."
        api_logger.log_tool_invocation(
            tool_name="get_reservation_details",
            args=tool_args,
            result=error_msg,
            success=False,
            error=str(e)
        )
        return error_msg

    except APIAuthenticationError as e:
        api_logger.logger.error(f"Authentication error retrieving reservation: {e}")
        error_msg = "I encountered an authentication error with the hotel system. Please contact support to verify the system configuration."
        api_logger.log_tool_invocation(
            tool_name="get_reservation_details",
            args=tool_args,
            result=error_msg,
            success=False,
            error=str(e)
        )
        return error_msg

    except APITimeoutError as e:
        api_logger.logger.error(f"Timeout retrieving reservation: {e}")
        error_msg = "The request took too long to complete. Please try again in a moment."
        api_logger.log_tool_invocation(
            tool_name="get_reservation_details",
            args=tool_args,
            result=error_msg,
            success=False,
            error=str(e)
        )
        return error_msg

    except APIConnectionError as e:
        api_logger.logger.error(f"Connection error retrieving reservation: {e}")
        error_msg = "I'm having trouble connecting to the hotel system right now. Please try again shortly."
        api_logger.log_tool_invocation(
            tool_name="get_reservation_details",
            args=tool_args,
            result=error_msg,
            success=False,
            error=str(e)
        )
        return error_msg

    except APIServerError as e:
        api_logger.logger.error(f"Server error retrieving reservation: {e}")
        error_msg = "The hotel system is experiencing issues at the moment. Please try again later or contact the front desk."
        api_logger.log_tool_invocation(
            tool_name="get_reservation_details",
            args=tool_args,
            result=error_msg,
            success=False,
            error=str(e)
        )
        return error_msg

    except Exception as e:
        api_logger.logger.error(f"Unexpected error retrieving reservation: {e}", exc_info=True)
        error_msg = "I encountered an unexpected error while retrieving the reservation. Please contact the front desk for assistance."
        api_logger.log_tool_invocation(
            tool_name="get_reservation_details",
            args=tool_args,
            result=error_msg,
            success=False,
            error=str(e)
        )
        return error_msg
