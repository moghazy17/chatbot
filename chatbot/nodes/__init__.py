"""
Audio Processing Nodes.

This module provides speech-to-text and text-to-speech nodes.
"""

from .speech_to_text import SpeechToTextNode, transcribe_audio
from .text_to_speech import TextToSpeechNode, synthesize_speech

__all__ = [
    "SpeechToTextNode",
    "transcribe_audio",
    "TextToSpeechNode",
    "synthesize_speech",
]
