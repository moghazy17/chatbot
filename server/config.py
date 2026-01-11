"""
Server Configuration.

Centralized configuration for the FastAPI server.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ServerConfig:
    """Server configuration."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # CORS settings
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    
    # LLM defaults
    default_provider: str = "ollama"
    default_model: Optional[str] = None
    default_temperature: float = 0.7
    
    # API keys (loaded from env)
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    
    # TTS settings
    default_tts_voice: str = "alloy"
    
    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Load configuration from environment variables."""
        return cls(
            host=os.getenv("SERVER_HOST", "0.0.0.0"),
            port=int(os.getenv("SERVER_PORT", "8000")),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            default_provider=os.getenv("LLM_PROVIDER", "ollama"),
            default_model=os.getenv("LLM_MODEL"),
            default_temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            default_tts_voice=os.getenv("TTS_VOICE", "alloy"),
        )
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider."""
        provider_lower = provider.lower()
        if provider_lower == "openai":
            return self.openai_api_key
        elif provider_lower == "groq":
            return self.groq_api_key
        return None


# Global config instance
config = ServerConfig.from_env()
