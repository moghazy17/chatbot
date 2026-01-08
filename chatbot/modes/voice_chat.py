"""
Voice Chat Handler.

Handles voice interactions with:
1. Record audio
2. Transcribe to text (STT)
3. Get AI response via LangGraph
4. Convert response to speech (TTS)
"""

from typing import Optional, Tuple

from .base import BaseChatHandler, ChatMessage, ChatMode
from .text_chat import TextChatHandler
from ..nodes import transcribe_audio, synthesize_speech


class VoiceChatHandler(BaseChatHandler):
    """
    Voice-based chat handler.
    
    Combines STT → LangGraph → TTS for voice conversations.
    Uses TextChatHandler internally for the LLM processing.
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: Optional[str] = None,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        tts_voice: str = "alloy",
        tts_model: str = "tts-1",
        stt_language: Optional[str] = None,
    ):
        """
        Initialize the voice chat handler.
        
        Args:
            provider: LLM provider
            model: Model name
            temperature: Sampling temperature
            api_key: API key for cloud providers
            base_url: Custom base URL
            tts_voice: Voice for text-to-speech
            tts_model: TTS model ("tts-1" or "tts-1-hd")
            stt_language: Language code for transcription
        """
        super().__init__(provider, model, temperature, api_key, base_url)
        
        self.tts_voice = tts_voice
        self.tts_model = tts_model
        self.stt_language = stt_language
        
        # Internal text handler for LLM processing
        self._text_handler: Optional[TextChatHandler] = None

    @property
    def mode(self) -> ChatMode:
        return ChatMode.VOICE

    def initialize(self) -> bool:
        """Initialize the underlying text chat handler."""
        try:
            self._text_handler = TextChatHandler(
                provider=self.provider,
                model=self.model,
                temperature=self.temperature,
                api_key=self.api_key,
                base_url=self.base_url,
            )
            success = self._text_handler.initialize()
            self._initialized = success
            return success
        except Exception as e:
            print(f"Failed to initialize VoiceChatHandler: {e}")
            self._initialized = False
            return False

    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio_bytes: Audio data
            filename: Filename with extension for format detection
            
        Returns:
            Transcribed text
        """
        return transcribe_audio(
            audio_bytes,
            language=self.stt_language,
            filename=filename,
        )

    def speak(self, text: str) -> Tuple[bytes, str]:
        """
        Convert text to speech.
        
        Args:
            text: Text to speak
            
        Returns:
            Tuple of (audio_bytes, format)
        """
        return synthesize_speech(
            text,
            model=self.tts_model,
            voice=self.tts_voice,
        )

    def send_message(self, content: str) -> ChatMessage:
        """
        Send a text message through the voice handler.
        
        Args:
            content: Message text
            
        Returns:
            Assistant's response
        """
        if not self._initialized or not self._text_handler:
            raise RuntimeError("Handler not initialized. Call initialize() first.")

        # Use the internal text handler
        response = self._text_handler.send_message(content)
        
        # Sync our messages with the text handler
        self._messages = self._text_handler._messages.copy()
        
        return response

    def process_audio(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        generate_audio: bool = True,
    ) -> Tuple[ChatMessage, Optional[Tuple[bytes, str]]]:
        """
        Process audio input: transcribe, get response, optionally generate speech.
        
        Args:
            audio_bytes: Recorded audio data
            filename: Audio filename for format detection
            generate_audio: Whether to generate TTS response
            
        Returns:
            Tuple of (response_message, optional (audio_bytes, format))
        """
        if not self._initialized:
            raise RuntimeError("Handler not initialized. Call initialize() first.")

        # Step 1: Transcribe
        transcribed_text = self.transcribe(audio_bytes, filename)
        
        if not transcribed_text:
            raise ValueError("Could not transcribe audio")

        # Step 2: Get AI response via text handler
        response_msg = self.send_message(transcribed_text)

        # Step 3: Generate speech (optional)
        audio_response = None
        if generate_audio and response_msg.content:
            audio_response = self.speak(response_msg.content)

        return response_msg, audio_response

    def clear_history(self) -> None:
        """Clear conversation history."""
        super().clear_history()
        if self._text_handler:
            self._text_handler.clear_history()

    @property
    def messages(self):
        """Get messages from the text handler for consistency."""
        if self._text_handler:
            return self._text_handler._messages
        return self._messages
