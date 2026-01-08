"""
Speech-to-Text Node using OpenAI Whisper API.

This module provides functionality to convert audio to text.
"""

import io
from typing import Optional

from ..openai_client import get_openai_client


class SpeechToTextNode:
    """
    Speech-to-Text node using OpenAI's Whisper API.

    Supports audio formats: wav, mp3, mp4, m4a, webm, flac, ogg
    """

    def __init__(
        self,
        model: str = "whisper-1",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        """
        Initialize the STT node.

        Args:
            model: Whisper model to use (default: whisper-1).
            language: Optional language code (e.g., 'en', 'es').
            prompt: Optional prompt to guide transcription.
        """
        self.model = model
        self.language = language
        self.prompt = prompt
        self.client = get_openai_client()

    def transcribe(self, audio_data: bytes, filename: str = "audio.wav") -> str:
        """
        Convert audio to text.

        Args:
            audio_data: Audio file bytes.
            filename: Filename with extension for format detection.

        Returns:
            Transcribed text string.
        """
        if not audio_data:
            return ""

        # Prepare audio file for API
        audio_file = io.BytesIO(audio_data)
        audio_file.name = filename

        # Build transcription parameters
        params = {
            "model": self.model,
            "file": audio_file,
            "response_format": "text",
        }

        if self.language:
            params["language"] = self.language
        if self.prompt:
            params["prompt"] = self.prompt

        # Call Whisper API
        transcription = self.client.audio.transcriptions.create(**params)

        return transcription

    def __call__(self, audio_data: bytes, filename: str = "audio.wav") -> str:
        """Callable interface for the node."""
        return self.transcribe(audio_data, filename)


def transcribe_audio(
    audio_data: bytes,
    model: str = "whisper-1",
    language: Optional[str] = None,
    filename: str = "audio.wav",
) -> str:
    """
    Convenience function to transcribe audio.

    Args:
        audio_data: Audio file bytes.
        model: Whisper model to use.
        language: Optional language code.
        filename: Filename with extension.

    Returns:
        Transcribed text string.
    """
    node = SpeechToTextNode(model=model, language=language)
    return node.transcribe(audio_data, filename)
