"""
Logging configuration for API operations.

Provides detailed logging for debugging API calls and tool invocations.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class APILogger:
    """Logger for API operations with file and console output."""

    def __init__(self, name: str):
        """
        Initialize API logger.

        Args:
            name: Logger name (usually module name)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Create logs directory if it doesn't exist
        logs_dir = Path(__file__).parent.parent.parent / "logs"
        logs_dir.mkdir(exist_ok=True)

        # File handler for detailed logs
        log_file = logs_dir / f"api_calls_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Detailed formatter for file
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)

        # Simpler formatter for console
        console_formatter = logging.Formatter(
            '%(levelname)s - %(name)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Tool invocation log file
        self.tool_log_file = logs_dir / f"tool_invocations_{datetime.now().strftime('%Y%m%d')}.log"

    def log_tool_invocation(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: str,
        success: bool = True,
        error: str = None
    ):
        """
        Log tool invocation details to both console and file.

        Args:
            tool_name: Name of the tool being invoked
            args: Tool arguments
            result: Tool result/response
            success: Whether the tool call succeeded
            error: Error message if failed
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Create detailed log entry
        log_entry = f"\n{'='*80}\n"
        log_entry += f"TOOL INVOCATION: {tool_name}\n"
        log_entry += f"Timestamp: {timestamp}\n"
        log_entry += f"Status: {'SUCCESS' if success else 'FAILURE'}\n"
        log_entry += f"\nArguments:\n"
        for key, value in args.items():
            log_entry += f"  {key}: {value}\n"
        log_entry += f"\nResult:\n{result}\n"
        if error:
            log_entry += f"\nError:\n{error}\n"
        log_entry += f"{'='*80}\n"

        # Write to tool invocation log file
        with open(self.tool_log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)

        # Also log to main logger
        if success:
            self.logger.info(f"Tool '{tool_name}' invoked successfully")
            self.logger.debug(f"Tool '{tool_name}' args: {args}")
        else:
            self.logger.error(f"Tool '{tool_name}' failed: {error}")

    def log_api_request(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        data: Any = None
    ):
        """
        Log API request details.

        Args:
            method: HTTP method
            url: Request URL
            headers: Request headers (sensitive data will be redacted)
            data: Request body
        """
        # Redact sensitive headers
        safe_headers = self._redact_sensitive_data(headers)

        self.logger.info(f"API Request: {method} {url}")
        self.logger.debug(f"Headers: {safe_headers}")
        if data:
            self.logger.debug(f"Request body: {data}")

    def log_api_response(
        self,
        status_code: int,
        response: Any,
        duration: float = None
    ):
        """
        Log API response details.

        Args:
            status_code: HTTP status code
            response: Response data
            duration: Request duration in seconds
        """
        duration_str = f" ({duration:.2f}s)" if duration else ""
        self.logger.info(f"API Response: {status_code}{duration_str}")
        self.logger.debug(f"Response body: {response}")

    def _redact_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Redact sensitive data from dictionaries.

        Args:
            data: Dictionary potentially containing sensitive data

        Returns:
            Dictionary with sensitive values redacted
        """
        sensitive_keys = ['bearer', 'token', 'password', 'api_key', 'authorization', 'app-key']
        redacted = data.copy()

        for key in list(redacted.keys()):
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                redacted[key] = '***REDACTED***'

        return redacted


def get_logger(name: str) -> APILogger:
    """
    Get or create an API logger.

    Args:
        name: Logger name

    Returns:
        APILogger instance
    """
    return APILogger(name)
