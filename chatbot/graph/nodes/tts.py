"""
Text-to-Speech Node.

Converts the assistant's text response to audio using OpenAI TTS API.
"""

from typing import Literal
from langchain_core.messages import AIMessage
from ..state import UnifiedChatState
from ...openai_client import get_openai_client


def should_generate_audio(state: UnifiedChatState) -> Literal["tts", "end"]:
    """
    Determine if we should generate audio output.
    
    Args:
        state: Current graph state
        
    Returns:
        "tts" if audio output requested, "end" otherwise
    """
    if state.get("output_type") == "audio":
        return "tts"
    return "end"


def tts_node(state: UnifiedChatState) -> dict:
    """
    Convert the last assistant message to speech.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with audio_output
    """
    messages = state.get("messages", [])
    
    # Find the last AI message with content
    response_text = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            response_text = msg.content
            break
    
    if not response_text:
        return {"audio_output": None}
    
    try:
        client = get_openai_client()
        
        # Get voice from llm_config or use default
        llm_config = state.get("llm_config") or {}
        voice = llm_config.get("tts_voice", "alloy")
        
        # Generate speech
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=response_text,
        )
        
        audio_bytes = response.content
        
        print(f"🔊 TTS: Generated audio ({len(audio_bytes)} bytes)")
        
        return {"audio_output": audio_bytes}
        
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        return {"audio_output": None}
