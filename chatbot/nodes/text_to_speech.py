"""
Text-to-Speech Node using OpenAI TTS API.

This module provides functionality to convert text to audio.
"""

from typing import Literal

from ..openai_client import get_openai_client


# Available TTS voices
TTSVoice = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

# Available TTS models
TTSModel = Literal["tts-1", "tts-1-hd"]

# Available output formats
TTSFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


class TextToSpeechNode:
    """
    Text-to-Speech node using OpenAI's TTS API.

    Available voices: alloy, echo, fable, onyx, nova, shimmer
    Available models: tts-1 (fast), tts-1-hd (high quality)
    """

    VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    def __init__(
        self,
        model: TTSModel = "tts-1",
        voice: TTSVoice = "alloy",
        response_format: TTSFormat = "mp3",
        speed: float = 1.0,
    ):
        """
        Initialize the TTS node.

        Args:
            model: TTS model ('tts-1' for speed, 'tts-1-hd' for quality).
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer).
            response_format: Audio format (mp3, opus, aac, flac, wav, pcm).
            speed: Speech speed (0.25 to 4.0, default 1.0).
        """
        self.model = model
        self.voice = voice
        self.response_format = response_format
        self.speed = max(0.25, min(4.0, speed))  # Clamp to valid range
        self.client = get_openai_client()

    def synthesize(self, text: str) -> tuple[bytes, str]:
        """
        Convert text to speech audio.

        Args:
            text: Text to convert to speech.

        Returns:
            Tuple of (audio_bytes, format_string).
        """
        if not text:
            return b"", self.response_format

        # Call TTS API
        response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format=self.response_format,
            speed=self.speed,
        )

        # Get audio bytes
        audio_bytes = response.content

        return audio_bytes, self.response_format

    def __call__(self, text: str) -> tuple[bytes, str]:
        """Callable interface for the node."""
        return self.synthesize(text)


def synthesize_speech(
    text: str,
    model: TTSModel = "tts-1",
    voice: TTSVoice = "alloy",
    response_format: TTSFormat = "mp3",
    speed: float = 1.0,
) -> tuple[bytes, str]:
    """
    Convenience function to synthesize speech.

    Args:
        text: Text to convert to speech.
        model: TTS model to use.
        voice: Voice to use.
        response_format: Output audio format.
        speed: Speech speed.

    Returns:
        Tuple of (audio_bytes, format_string).
    """
    node = TextToSpeechNode(
        model=model,
        voice=voice,
        response_format=response_format,
        speed=speed,
    )
    return node.synthesize(text)
