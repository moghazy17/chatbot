"""
Speech-to-Text Node.

Transcribes audio input to text using OpenAI Whisper API.
"""

from langchain_core.messages import HumanMessage
from ..state import UnifiedChatState
from ...openai_client import get_openai_client


def stt_node(state: UnifiedChatState) -> dict:
    """
    Transcribe audio input to text.
    
    Args:
        state: Current graph state with audio_input
        
    Returns:
        Updated state with transcription and human message added
    """
    audio_input = state.get("audio_input")
    
    if not audio_input:
        return {"transcription": None}
    
    try:
        client = get_openai_client()
        
        # Create a file-like object for the API
        import io
        audio_file = io.BytesIO(audio_input)
        audio_file.name = "audio.wav"
        
        # Transcribe using Whisper
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )
        
        transcription = response.text
        
        print(f"🎤 STT: Transcribed audio -> '{transcription}'")
        
        # Add transcription as a human message
        return {
            "transcription": transcription,
            "messages": [HumanMessage(content=transcription)],
        }
        
    except Exception as e:
        print(f"❌ STT Error: {e}")
        return {"transcription": None}
