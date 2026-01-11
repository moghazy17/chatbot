"""
System Prompt Node.

Adds the system prompt to the conversation if not already present.
"""

from langchain_core.messages import SystemMessage
from ..state import UnifiedChatState


# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. You have access to tools that can help you answer questions.

IMPORTANT RULES FOR TOOL USAGE:
1. Only use tools when the user's request REQUIRES information you don't have or an action you can't perform directly.
2. For simple greetings (hi, hello, how are you), casual conversation, or questions you can answer from your knowledge - DO NOT use tools. Just respond naturally.
3. Use tools ONLY when:
   - The user asks about specific hotel reservations, bookings, or guest information
   - The user needs real-time data you cannot provide from memory
   - The user explicitly asks you to perform an action (create, update, delete something)

If you're unsure whether to use a tool, prefer responding directly first."""


def prompt_node(state: UnifiedChatState) -> dict:
    """
    Add system prompt to messages if not present.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with system prompt added if needed
    """
    messages = list(state.get("messages", []))
    
    # Check if system message already exists
    has_system_message = any(
        isinstance(msg, SystemMessage) for msg in messages
    )
    
    if not has_system_message:
        # Prepend system message
        system_msg = SystemMessage(content=DEFAULT_SYSTEM_PROMPT)
        return {"messages": [system_msg]}
    
    # No changes needed
    return {}
