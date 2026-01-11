"""
Chat Router.

Handles chat endpoints for text and voice interactions.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..repository.chat_repository import ChatRepository
from ..config import config

router = APIRouter()


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


@router.post("/voice", response_model=VoiceChatResponse)
async def voice_chat(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    output_audio: bool = Form(True),
    provider: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
):
    """
    Send audio and get a response.
    
    Args:
        audio: Audio file upload
        session_id: Optional session ID
        output_audio: Whether to generate audio response
        provider: Optional LLM provider override
        model: Optional model override
        
    Returns:
        VoiceChatResponse with transcription and response
    """
    repo = get_chat_repo()
    
    try:
        # Read audio bytes
        audio_bytes = await audio.read()
        
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
