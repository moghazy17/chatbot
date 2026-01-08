"""
OpenAI Client Module.

This module provides a singleton OpenAI client for Whisper STT and TTS APIs.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class OpenAIClient:
    """Singleton OpenAI client for audio processing."""

    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_client()
        return cls._instance

    def _initialize_client(self):
        """Initialize the OpenAI client with API key."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        # Strip any whitespace from the API key
        api_key = api_key.strip()
        self._client = OpenAI(api_key=api_key)

    @property
    def client(self) -> OpenAI:
        """Get the OpenAI client instance."""
        return self._client


def get_openai_client() -> OpenAI:
    """
    Get the singleton OpenAI client.

    Returns:
        OpenAI client instance.

    Raises:
        ValueError: If OPENAI_API_KEY is not set.
    """
    return OpenAIClient().client
