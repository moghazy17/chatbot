"""
Unified Chat State Definition.

Defines the state schema used by the LangGraph for processing
both text and voice inputs through a single unified graph.
"""

from typing import Annotated, Sequence, Optional, Literal, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class UnifiedChatState(TypedDict):
    """
    Unified state for the chatbot graph.
    
    Supports both text and audio input/output with conditional routing.
    
    Attributes:
        messages: Conversation history (LangChain messages)
        input_type: Type of input - "text" or "audio" (routing key)
        output_type: Desired output format - "text" or "audio"
        audio_input: Raw audio bytes if input_type is "audio"
        audio_output: Generated audio bytes if output_type is "audio"
        transcription: Transcribed text from audio input
        llm_config: LLM configuration (provider, model, etc.)
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    input_type: Literal["text", "audio"]
    output_type: Literal["text", "audio"]
    audio_input: Optional[bytes]
    audio_output: Optional[bytes]
    transcription: Optional[str]
    llm_config: Optional[dict[str, Any]]


def create_initial_state(
    input_type: Literal["text", "audio"] = "text",
    output_type: Literal["text", "audio"] = "text",
    llm_config: Optional[dict[str, Any]] = None,
) -> UnifiedChatState:
    """
    Create an initial state for the graph.
    
    Args:
        input_type: Type of input ("text" or "audio")
        output_type: Desired output format ("text" or "audio")
        llm_config: Optional LLM configuration
        
    Returns:
        Initial UnifiedChatState
    """
    return UnifiedChatState(
        messages=[],
        input_type=input_type,
        output_type=output_type,
        audio_input=None,
        audio_output=None,
        transcription=None,
        llm_config=llm_config,
    )
