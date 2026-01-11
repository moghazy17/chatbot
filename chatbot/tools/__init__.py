"""
Tools Module.

Contains all LLM tools for the chatbot.
Tools auto-register on import via the @tools_registry.register decorator.
"""

from .registry import ToolsRegistry, tools_registry

# Import tool modules to trigger auto-registration
try:
    from .oracle_hospitality import get_reservation_details, create_service_request
except ImportError as e:
    print(f"Warning: Could not load Oracle Hospitality tools: {e}")
except ValueError as e:
    print(f"Warning: Oracle API not configured: {e}")

__all__ = [
    "ToolsRegistry",
    "tools_registry",
]
