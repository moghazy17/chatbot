"""
Chat Router.

Handles chat endpoints for text and voice interactions.
"""

import os
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..repository.chat_repository import ChatRepository
from ..config import config

router = APIRouter(prefix="/api/v1/chat-inhouse")

# Supported audio formats for Whisper API
SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".mp4", ".m4a", ".flac", 
    ".ogg", ".webm", ".mpeg", ".mpga"
}
SUPPORTED_AUDIO_MIME_TYPES = {
    "audio/wav", "audio/wave", "audio/x-wav",
    "audio/mpeg", "audio/mp3", "audio/mpeg3",
    "audio/mp4", "audio/m4a", "audio/x-m4a",
    "audio/flac", "audio/x-flac",
    "audio/ogg", "audio/vorbis",
    "audio/webm",
    "video/mp4", "video/webm",  # Some containers have audio
}


class ChatRequest(BaseModel):
    """Request model for text chat."""
    message: str
    session_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None


class ChatResponse(BaseModel):
    """Response model for chat."""
    response: str
    session_id: str


class VoiceChatResponse(BaseModel):
    """Response model for voice chat."""
    response: str
    transcription: str
    session_id: str
    audio_base64: Optional[str] = None


# Repository instance
_chat_repo: Optional[ChatRepository] = None


def get_chat_repo() -> ChatRepository:
    """Get or create the chat repository."""
    global _chat_repo
    if _chat_repo is None:
        _chat_repo = ChatRepository(
            provider=config.default_provider,
            model=config.default_model,
            temperature=config.default_temperature,
            api_key=config.openai_api_key or config.groq_api_key,
        )
    return _chat_repo


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a text message and get a response.
    
    Args:
        request: ChatRequest with message and optional config
        
    Returns:
        ChatResponse with the assistant's response
    """
    repo = get_chat_repo()
    
    try:
        response, session_id = repo.send_text(
            message=request.message,
            session_id=request.session_id,
            provider=request.provider,
            model=request.model,
            temperature=request.temperature,
        )
        
        return ChatResponse(
            response=response,
            session_id=session_id,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/voice",
    response_model=VoiceChatResponse,
    summary="Send audio and get voice response",
    description="""
    Send an audio file and receive a transcribed response with optional audio output.
    
    **Supported Audio Formats:**
    - WAV (`.wav`) - Recommended, uncompressed
    - MP3 (`.mp3`) - Common compressed format
    - MP4/M4A (`.mp4`, `.m4a`) - Video/audio container
    - FLAC (`.flac`) - Lossless compression
    - OGG (`.ogg`) - Open format
    - WebM (`.webm`) - Web-optimized
    - MPEG (`.mpeg`, `.mpga`) - Legacy formats
    
    **Requirements:**
    - Max file size: 25MB (OpenAI Whisper limit)
    - Audio should contain clear speech for best transcription
    - Requires `OPENAI_API_KEY` to be set (for Whisper STT and TTS)
    
    **Response:**
    - `transcription`: What the user said (transcribed from audio)
    - `response`: AI's text response
    - `audio_base64`: AI's audio response (MP3 format, base64 encoded) if `output_audio=true`
    - `session_id`: Session ID for conversation continuity
    """,
    responses={
        200: {
            "description": "Successfully processed audio and generated response",
            "content": {
                "application/json": {
                    "example": {
                        "response": "Hello! How can I help you today?",
                        "transcription": "Hello",
                        "session_id": "abc-123-def-456",
                        "audio_base64": "UklGRiQAAABXQVZFZm10..."
                    }
                }
            }
        },
        400: {
            "description": "Invalid audio format or file too large",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Unsupported audio format. Supported formats: .wav, .mp3, .mp4, .m4a, .flac, .ogg, .webm, .mpeg, .mpga"
                    }
                }
            }
        },
        500: {
            "description": "Server error during processing",
        }
    }
)
async def voice_chat(
    audio: UploadFile = File(
        ...,
        description="Audio file to transcribe. Supported formats: WAV, MP3, MP4, M4A, FLAC, OGG, WebM, MPEG",
    ),
    session_id: Optional[str] = Form(
        None,
        description="Optional session ID for conversation continuity. If not provided, a new session will be created."
    ),
    output_audio: bool = Form(
        True,
        description="Whether to generate audio response (TTS). If false, only text response is returned."
    ),
    provider: Optional[str] = Form(
        None,
        description="Optional LLM provider override (ollama, groq, openai). Uses server default if not specified."
    ),
    model: Optional[str] = Form(
        None,
        description="Optional model override. Uses provider default if not specified."
    ),
):
    """
    Send audio and get a response.
    
    Validates audio format and processes through the unified graph.
    """
    repo = get_chat_repo()
    
    # Validate audio file extension
    if audio.filename:
        file_ext = os.path.splitext(audio.filename.lower())[1]
        if file_ext not in SUPPORTED_AUDIO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format '{file_ext}'. Supported formats: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}"
            )
    
    # Validate MIME type if available
    if audio.content_type:
        # Check if it's a supported audio/video MIME type
        is_supported = (
            audio.content_type.startswith("audio/") or
            audio.content_type.startswith("video/") or
            audio.content_type in SUPPORTED_AUDIO_MIME_TYPES
        )
        if not is_supported:
            # Don't reject based on MIME type alone, but log a warning
            print(f"⚠️ Warning: Unexpected MIME type '{audio.content_type}' for file '{audio.filename}'")
    
    # Check file size (25MB limit for OpenAI Whisper)
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
    
    try:
        # Read audio bytes
        audio_bytes = await audio.read()
        
        if len(audio_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Audio file too large. Maximum size is 25MB, received {len(audio_bytes) / 1024 / 1024:.2f}MB"
            )
        
        if len(audio_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail="Audio file is empty"
            )
        
        response, transcription, session_id, audio_output = repo.send_audio(
            audio_bytes=audio_bytes,
            session_id=session_id,
            generate_audio=output_audio,
            provider=provider,
            model=model,
        )
        
        # Encode audio to base64 if present
        audio_base64 = None
        if audio_output:
            import base64
            audio_base64 = base64.b64encode(audio_output).decode("utf-8")
        
        return VoiceChatResponse(
            response=response,
            transcription=transcription,
            session_id=session_id,
            audio_base64=audio_base64,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def stream_chat(request: ChatRequest):
    """
    Stream a text chat response.
    
    Args:
        request: ChatRequest with message
        
    Returns:
        StreamingResponse with chunks
    """
    repo = get_chat_repo()
    
    async def generate():
        try:
            for chunk in repo.stream_text(
                message=request.message,
                session_id=request.session_id,
                provider=request.provider,
                model=request.model,
                temperature=request.temperature,
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )


@router.delete("/{session_id}")
async def clear_session(session_id: str):
    """
    Clear a chat session.
    
    Args:
        session_id: Session ID to clear
        
    Returns:
        Confirmation message
    """
    repo = get_chat_repo()
    repo.clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}
