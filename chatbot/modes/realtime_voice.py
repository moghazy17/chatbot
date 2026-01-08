"""
Realtime Voice Handler.

Provides continuous real-time voice conversation using OpenAI's Realtime API.
This mode maintains an always-listening connection with server-side voice detection.
"""

import os
import queue
import threading
import time
from typing import Optional, Callable, List
from dataclasses import dataclass

from .base import BaseChatHandler, ChatMessage, ChatMode
from ..realtime_client import RealtimeClient, RealtimeConfig


@dataclass
class RealtimeCallbacks:
    """Callbacks for realtime voice events."""
    on_user_speech_start: Optional[Callable[[], None]] = None
    on_user_speech_end: Optional[Callable[[str], None]] = None
    on_assistant_speech_start: Optional[Callable[[], None]] = None
    on_assistant_speech_end: Optional[Callable[[str], None]] = None
    on_audio_chunk: Optional[Callable[[bytes], None]] = None
    on_transcript_delta: Optional[Callable[[str, str], None]] = None
    on_error: Optional[Callable[[str], None]] = None
    on_connection_change: Optional[Callable[[bool], None]] = None


class RealtimeVoiceHandler(BaseChatHandler):
    """
    Real-time voice conversation handler.
    
    Uses OpenAI's Realtime API for continuous, low-latency voice interactions.
    Features server-side VAD (Voice Activity Detection) for hands-free operation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice: str = "alloy",
        instructions: Optional[str] = None,
        callbacks: Optional[RealtimeCallbacks] = None,
        vad_threshold: float = 0.5,
        silence_duration_ms: int = 500,
    ):
        """
        Initialize the realtime voice handler.
        
        Args:
            api_key: OpenAI API key (uses env var if not provided)
            voice: Voice for the assistant
            instructions: System instructions
            callbacks: Event callbacks
            vad_threshold: Voice activity detection sensitivity
            silence_duration_ms: Silence duration to end turn
        """
        # Don't need LLM provider settings - uses OpenAI Realtime
        super().__init__()
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
        self.voice = voice
        self.instructions = instructions or (
            "You are a helpful assistant. Respond naturally and conversationally. "
            "Keep responses concise for voice interaction."
        )
        self.callbacks = callbacks or RealtimeCallbacks()
        self.vad_threshold = vad_threshold
        self.silence_duration_ms = silence_duration_ms
        
        self._client: Optional[RealtimeClient] = None
        self._audio_output_queue: queue.Queue = queue.Queue()
        self._current_user_transcript = ""
        self._current_assistant_transcript = ""
        self._is_assistant_speaking = False
        self._lock = threading.Lock()

    @property
    def mode(self) -> ChatMode:
        return ChatMode.REALTIME

    @property
    def is_connected(self) -> bool:
        """Check if connected to the Realtime API."""
        return self._client is not None and self._client.is_connected

    def initialize(self) -> bool:
        """
        Initialize the realtime connection.
        
        Note: This establishes the WebSocket connection to OpenAI.
        """
        try:
            if not self.api_key:
                raise ValueError("OpenAI API key is required for realtime voice")

            config = RealtimeConfig(
                voice=self.voice,
                instructions=self.instructions,
                turn_detection={
                    "type": "server_vad",
                    "threshold": self.vad_threshold,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": self.silence_duration_ms,
                },
            )

            self._client = RealtimeClient(
                api_key=self.api_key,
                config=config,
                on_audio=self._handle_audio_output,
                on_transcript=self._handle_transcript,
                on_error=self._handle_error,
            )

            self._client.start()
            
            # Wait for connection
            timeout = 5.0
            start = time.time()
            while not self._client.is_connected and (time.time() - start) < timeout:
                time.sleep(0.1)

            self._initialized = self._client.is_connected
            
            if self._initialized and self.callbacks.on_connection_change:
                self.callbacks.on_connection_change(True)

            return self._initialized

        except Exception as e:
            print(f"Failed to initialize RealtimeVoiceHandler: {e}")
            if self.callbacks.on_error:
                self.callbacks.on_error(str(e))
            self._initialized = False
            return False

    def _handle_audio_output(self, audio_bytes: bytes):
        """Handle audio chunk from the API."""
        self._audio_output_queue.put(audio_bytes)
        
        if not self._is_assistant_speaking:
            self._is_assistant_speaking = True
            if self.callbacks.on_assistant_speech_start:
                self.callbacks.on_assistant_speech_start()
        
        if self.callbacks.on_audio_chunk:
            self.callbacks.on_audio_chunk(audio_bytes)

    def _handle_transcript(self, role: str, text: str):
        """Handle transcript update from the API."""
        with self._lock:
            if role == "user":
                self._current_user_transcript += text
                if self.callbacks.on_transcript_delta:
                    self.callbacks.on_transcript_delta("user", text)
            else:
                self._current_assistant_transcript += text
                if self.callbacks.on_transcript_delta:
                    self.callbacks.on_transcript_delta("assistant", text)

    def _handle_error(self, error: str):
        """Handle error from the API."""
        print(f"Realtime API error: {error}")
        if self.callbacks.on_error:
            self.callbacks.on_error(error)

    def send_audio(self, audio_bytes: bytes):
        """
        Send audio data to the API.
        
        Args:
            audio_bytes: PCM16 audio data at 24kHz mono
        """
        if self._client and self._client.is_connected:
            self._client.send_audio(audio_bytes)

    def send_message(self, content: str) -> ChatMessage:
        """
        Send a text message through the realtime connection.
        
        Args:
            content: Message text
            
        Returns:
            ChatMessage (note: response comes through callbacks)
        """
        if not self._initialized or not self._client:
            raise RuntimeError("Handler not initialized. Call initialize() first.")

        # Add to history
        self.add_message("user", content)
        
        # Send via realtime client
        self._client.send_text(content)
        
        # Return placeholder - actual response comes via callbacks
        return ChatMessage(role="user", content=content)

    def get_audio_output(self, timeout: float = 0.1) -> Optional[bytes]:
        """
        Get next audio chunk from output queue.
        
        Args:
            timeout: How long to wait for audio
            
        Returns:
            Audio bytes or None if no audio available
        """
        try:
            return self._audio_output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def flush_user_transcript(self) -> Optional[str]:
        """
        Get and clear the current user transcript.
        
        Returns:
            User's transcribed speech or None if empty
        """
        with self._lock:
            transcript = self._current_user_transcript.strip()
            if transcript:
                self.add_message("user", transcript)
                self._current_user_transcript = ""
                if self.callbacks.on_user_speech_end:
                    self.callbacks.on_user_speech_end(transcript)
                return transcript
        return None

    def flush_assistant_transcript(self) -> Optional[str]:
        """
        Get and clear the current assistant transcript.
        
        Returns:
            Assistant's transcribed speech or None if empty
        """
        with self._lock:
            transcript = self._current_assistant_transcript.strip()
            if transcript:
                self.add_message("assistant", transcript)
                self._current_assistant_transcript = ""
                self._is_assistant_speaking = False
                if self.callbacks.on_assistant_speech_end:
                    self.callbacks.on_assistant_speech_end(transcript)
                return transcript
        return None

    def stop(self):
        """Stop the realtime connection."""
        if self._client:
            self._client.stop()
            self._client = None
        
        self._initialized = False
        
        if self.callbacks.on_connection_change:
            self.callbacks.on_connection_change(False)

    def __del__(self):
        """Cleanup on destruction."""
        self.stop()
