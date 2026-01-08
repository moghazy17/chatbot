"""
OpenAI Realtime API Client.

This module provides a client for real-time voice conversations using
OpenAI's Realtime API with WebSocket streaming.
"""

import os
import json
import base64
import asyncio
import threading
from typing import Callable, Optional
from dataclasses import dataclass, field

import websockets
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RealtimeConfig:
    """Configuration for the Realtime API session."""

    model: str = "gpt-4o-realtime-preview-2024-12-17"
    voice: str = "alloy"  # alloy, echo, fable, onyx, nova, shimmer
    instructions: str = "You are a helpful assistant. Respond naturally and conversationally."
    input_audio_format: str = "pcm16"
    output_audio_format: str = "pcm16"
    input_audio_transcription: bool = True
    turn_detection: dict = field(default_factory=lambda: {
        "type": "server_vad",
        "threshold": 0.5,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 500,
    })
    temperature: float = 0.8
    max_response_output_tokens: int = 4096


class RealtimeClient:
    """
    Client for OpenAI Realtime API.

    Handles WebSocket connection for real-time voice conversations.
    """

    REALTIME_URL = "wss://api.openai.com/v1/realtime"

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[RealtimeConfig] = None,
        on_audio: Optional[Callable[[bytes], None]] = None,
        on_transcript: Optional[Callable[[str, str], None]] = None,
        on_transcript_done: Optional[Callable[[str, str], None]] = None,
        on_speech_started: Optional[Callable[[], None]] = None,
        on_speech_stopped: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize the Realtime client.

        Args:
            api_key: OpenAI API key. Uses OPENAI_API_KEY env var if not provided.
            config: Session configuration.
            on_audio: Callback for audio output (receives PCM bytes).
            on_transcript: Callback for transcript deltas (role, text_delta).
            on_transcript_done: Callback for completed transcripts (role, full_text).
            on_speech_started: Callback when AI starts speaking.
            on_speech_stopped: Callback when AI stops speaking.
            on_error: Callback for errors.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.config = config or RealtimeConfig()
        self.on_audio = on_audio
        self.on_transcript = on_transcript
        self.on_transcript_done = on_transcript_done
        self.on_speech_started = on_speech_started
        self.on_speech_stopped = on_speech_stopped
        self.on_error = on_error

        self._ws = None
        self._running = False
        self._loop = None
        self._thread = None
        self._assistant_transcript = ""

    def _get_url(self) -> str:
        """Get the WebSocket URL with model parameter."""
        return f"{self.REALTIME_URL}?model={self.config.model}"

    def _get_headers(self) -> dict:
        """Get WebSocket connection headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

    async def _connect(self):
        """Establish WebSocket connection."""
        try:
            self._ws = await websockets.connect(
                self._get_url(),
                additional_headers=self._get_headers(),
                ping_interval=20,
                ping_timeout=10,
            )
            await self._configure_session()
        except websockets.exceptions.InvalidStatusCode as e:
            error_msg = f"Connection failed (HTTP {e.status_code})"
            if e.status_code == 401:
                error_msg = "Invalid API key. Check your OPENAI_API_KEY in .env file."
            elif e.status_code == 403:
                error_msg = "Access denied. Your API key may not have Realtime API access."
            if self.on_error:
                self.on_error(error_msg)
            raise
        except Exception as e:
            if self.on_error:
                self.on_error(f"Connection error: {str(e)}")
            raise

    async def _configure_session(self):
        """Send session configuration."""
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": self.config.instructions,
                "voice": self.config.voice,
                "input_audio_format": self.config.input_audio_format,
                "output_audio_format": self.config.output_audio_format,
                "turn_detection": self.config.turn_detection,
                "temperature": self.config.temperature,
                "max_response_output_tokens": self.config.max_response_output_tokens,
            },
        }

        if self.config.input_audio_transcription:
            session_config["session"]["input_audio_transcription"] = {
                "model": "whisper-1"
            }

        await self._ws.send(json.dumps(session_config))

    async def _receive_loop(self):
        """Main loop to receive and process messages."""
        try:
            async for message in self._ws:
                await self._handle_message(json.loads(message))
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            if self.on_error:
                self.on_error(str(e))

    async def _handle_message(self, msg: dict):
        """Handle incoming WebSocket message."""
        msg_type = msg.get("type", "")

        if msg_type == "error":
            error_msg = msg.get("error", {}).get("message", "Unknown error")
            if self.on_error:
                self.on_error(error_msg)

        elif msg_type == "response.audio.delta":
            # Audio chunk received - AI is speaking
            audio_b64 = msg.get("delta", "")
            if audio_b64 and self.on_audio:
                audio_bytes = base64.b64decode(audio_b64)
                self.on_audio(audio_bytes)

        elif msg_type == "response.audio_transcript.delta":
            # Assistant transcript chunk (accumulate)
            text = msg.get("delta", "")
            if text:
                self._assistant_transcript += text
                if self.on_transcript:
                    self.on_transcript("assistant", text)

        elif msg_type == "response.audio_transcript.done":
            # Assistant finished speaking - send complete transcript
            full_text = msg.get("transcript", self._assistant_transcript)
            if full_text and self.on_transcript_done:
                self.on_transcript_done("assistant", full_text)
            self._assistant_transcript = ""
            if self.on_speech_stopped:
                self.on_speech_stopped()

        elif msg_type == "conversation.item.input_audio_transcription.completed":
            # User speech transcription (already complete)
            text = msg.get("transcript", "")
            if text:
                if self.on_transcript:
                    self.on_transcript("user", text)
                if self.on_transcript_done:
                    self.on_transcript_done("user", text)

        elif msg_type == "response.created":
            # AI starting to respond
            if self.on_speech_started:
                self.on_speech_started()

        elif msg_type == "response.done":
            # Response fully completed
            pass

        elif msg_type == "input_audio_buffer.speech_started":
            # User started speaking (VAD detected speech)
            pass

        elif msg_type == "input_audio_buffer.speech_stopped":
            # User stopped speaking (VAD detected silence)
            pass

        elif msg_type == "session.created":
            # Session established
            pass

        elif msg_type == "session.updated":
            # Session config updated
            pass

    async def _send_audio(self, audio_bytes: bytes):
        """Send audio data to the API."""
        if self._ws:
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            await self._ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": audio_b64,
            }))

    async def _commit_audio(self):
        """Commit the audio buffer to trigger processing."""
        if self._ws:
            await self._ws.send(json.dumps({
                "type": "input_audio_buffer.commit",
            }))
            await self._ws.send(json.dumps({
                "type": "response.create",
            }))

    async def _run(self):
        """Main async run loop."""
        try:
            await self._connect()
            self._running = True
            await self._receive_loop()
        except websockets.exceptions.ConnectionClosedError as e:
            # Extract error message from close reason
            error_msg = str(e)
            if "invalid_api_key" in error_msg.lower():
                error_msg = "Invalid API key. Please check your OPENAI_API_KEY in .env file."
            if self.on_error:
                self.on_error(error_msg)
            self._running = False
        except Exception as e:
            if self.on_error:
                self.on_error(f"Realtime API error: {str(e)}")
            self._running = False

    def _run_in_thread(self):
        """Run the event loop in a separate thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run())
        except Exception as e:
            error_msg = str(e)
            if "invalid_api_key" in error_msg.lower():
                error_msg = "Invalid API key. Please check your OPENAI_API_KEY."
            if self.on_error:
                self.on_error(error_msg)
            self._running = False
        finally:
            self._loop.close()

    def start(self):
        """Start the realtime connection in a background thread."""
        if self._thread and self._thread.is_alive():
            return

        self._thread = threading.Thread(target=self._run_in_thread, daemon=True)
        self._thread.start()

        # Wait for connection to establish
        import time
        for _ in range(50):  # 5 second timeout
            if self._running:
                break
            time.sleep(0.1)

    def stop(self):
        """Stop the realtime connection."""
        self._running = False
        if self._loop and self._ws:
            asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop)

    def send_audio(self, audio_bytes: bytes):
        """
        Send audio data to the API.

        Args:
            audio_bytes: PCM16 audio data at 24kHz mono.
        """
        if self._loop and self._running:
            asyncio.run_coroutine_threadsafe(
                self._send_audio(audio_bytes), self._loop
            )

    def commit_audio(self):
        """Commit audio buffer and request response."""
        if self._loop and self._running:
            asyncio.run_coroutine_threadsafe(
                self._commit_audio(), self._loop
            )

    def send_text(self, text: str):
        """
        Send a text message instead of audio.

        Args:
            text: Text message to send.
        """
        if self._loop and self._running:
            asyncio.run_coroutine_threadsafe(
                self._send_text_message(text), self._loop
            )

    async def _send_text_message(self, text: str):
        """Send text message to the API."""
        if self._ws:
            await self._ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }))
            await self._ws.send(json.dumps({
                "type": "response.create",
            }))

    @property
    def is_connected(self) -> bool:
        """Check if connected to the API."""
        return self._running and self._ws is not None


def create_realtime_client(
    voice: str = "alloy",
    instructions: str = None,
    on_audio: Callable[[bytes], None] = None,
    on_transcript: Callable[[str, str], None] = None,
    on_error: Callable[[str], None] = None,
) -> RealtimeClient:
    """
    Create a configured Realtime client.

    Args:
        voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer).
        instructions: System instructions for the assistant.
        on_audio: Callback for audio output.
        on_transcript: Callback for transcripts.
        on_error: Callback for errors.

    Returns:
        Configured RealtimeClient instance.
    """
    config = RealtimeConfig(
        voice=voice,
        instructions=instructions or RealtimeConfig.instructions,
    )
    return RealtimeClient(
        config=config,
        on_audio=on_audio,
        on_transcript=on_transcript,
        on_error=on_error,
    )
