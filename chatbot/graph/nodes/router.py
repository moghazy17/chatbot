"""
Input Router Node.

Routes the input based on input_type to either STT or directly to prompt processing.
"""

from typing import Literal
from ..state import UnifiedChatState


def route_by_input_type(state: UnifiedChatState) -> Literal["stt", "prompt"]:
    """
    Route based on input type.
    
    Args:
        state: Current graph state
        
    Returns:
        "stt" if audio input, "prompt" if text input
    """
    if state.get("input_type") == "audio" and state.get("audio_input"):
        return "stt"
    return "prompt"
