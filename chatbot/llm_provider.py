"""
LLM Provider Module.

Provides a unified interface for different LLM providers (Ollama, Groq, OpenAI).
"""

import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Any
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    GROQ = "groq"
    OPENAI = "openai"


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""
    provider: LLMProvider
    model: str
    temperature: float = 0.7
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """
        Load LLM configuration from environment variables.

        Environment variables:
            LLM_PROVIDER: Provider name (ollama, groq, openai)
            LLM_MODEL: Model name/ID
            LLM_TEMPERATURE: Sampling temperature (default: 0.7)
            OLLAMA_BASE_URL: Ollama server URL (for ollama provider)
            GROQ_API_KEY: Groq API key (for groq provider)
            OPENAI_API_KEY: OpenAI API key (for openai provider)

        Returns:
            LLMConfig instance
        """
        provider_str = os.getenv('LLM_PROVIDER', 'ollama').lower()

        try:
            provider = LLMProvider(provider_str)
        except ValueError:
            print(f"Warning: Unknown LLM provider '{provider_str}', defaulting to ollama")
            provider = LLMProvider.OLLAMA

        # Get model based on provider
        if provider == LLMProvider.OLLAMA:
            model = os.getenv('OLLAMA_MODEL', 'llama3.2')
            base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            api_key = None
        elif provider == LLMProvider.GROQ:
            model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
            api_key = os.getenv('GROQ_API_KEY')
            base_url = None
        elif provider == LLMProvider.OPENAI:
            model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
            api_key = os.getenv('OPENAI_API_KEY')
            base_url = os.getenv('OPENAI_BASE_URL')  # For Azure or custom endpoints
        else:
            model = os.getenv('LLM_MODEL', 'llama3.2')
            api_key = None
            base_url = None

        temperature = float(os.getenv('LLM_TEMPERATURE', '0.7'))

        return cls(
            provider=provider,
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url
        )


class LLMFactory:
    """Factory for creating LLM instances based on provider configuration."""

    # Default models for each provider
    DEFAULT_MODELS = {
        LLMProvider.OLLAMA: [
            "llama3.2",
            "llama3.1",
            "mistral",
            "codellama",
            "qwen3:4b",
            "phi3",
        ],
        LLMProvider.GROQ: [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama-3.2-90b-vision-preview",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        LLMProvider.OPENAI: [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "o1-mini",
            "o1-preview",
        ],
    }

    @staticmethod
    def create(
        provider: LLMProvider,
        model: str,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        tools: Optional[List[Any]] = None,
    ) -> BaseChatModel:
        """
        Create an LLM instance based on the provider.

        Args:
            provider: The LLM provider to use
            model: Model name/ID
            temperature: Sampling temperature
            api_key: API key (for Groq/OpenAI)
            base_url: Base URL (for Ollama or custom endpoints)
            tools: Optional list of tools to bind

        Returns:
            BaseChatModel instance

        Raises:
            ValueError: If provider is not supported or required credentials are missing
        """
        llm: BaseChatModel

        if provider == LLMProvider.OLLAMA:
            from langchain_ollama import ChatOllama

            llm = ChatOllama(
                model=model,
                temperature=temperature,
                base_url=base_url or "http://localhost:11434",
            )
            print(f"✓ Using Ollama with model: {model}")

        elif provider == LLMProvider.GROQ:
            if not api_key:
                raise ValueError(
                    "GROQ_API_KEY is required for Groq provider. "
                    "Please set it in your .env file."
                )

            from langchain_groq import ChatGroq

            llm = ChatGroq(
                model=model,
                temperature=temperature,
                api_key=api_key,
            )
            print(f"✓ Using Groq with model: {model}")

        elif provider == LLMProvider.OPENAI:
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required for OpenAI provider. "
                    "Please set it in your .env file."
                )

            from langchain_openai import ChatOpenAI

            kwargs = {
                "model": model,
                "temperature": temperature,
                "api_key": api_key,
            }
            if base_url:
                kwargs["base_url"] = base_url

            llm = ChatOpenAI(**kwargs)
            print(f"✓ Using OpenAI with model: {model}")

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        # Bind tools if provided
        if tools:
            llm = llm.bind_tools(tools)
            print(f"  → Bound {len(tools)} tool(s)")

        return llm

    @classmethod
    def create_from_config(
        cls,
        config: LLMConfig,
        tools: Optional[List[Any]] = None,
    ) -> BaseChatModel:
        """
        Create an LLM instance from configuration.

        Args:
            config: LLMConfig instance
            tools: Optional list of tools to bind

        Returns:
            BaseChatModel instance
        """
        return cls.create(
            provider=config.provider,
            model=config.model,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url,
            tools=tools,
        )

    @classmethod
    def get_default_models(cls, provider: LLMProvider) -> List[str]:
        """
        Get default model list for a provider.

        Args:
            provider: The LLM provider

        Returns:
            List of model names
        """
        return cls.DEFAULT_MODELS.get(provider, [])

    @classmethod
    def get_all_providers(cls) -> List[LLMProvider]:
        """Get all supported providers."""
        return list(LLMProvider)


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    tools: Optional[List[Any]] = None,
) -> BaseChatModel:
    """
    Convenience function to get an LLM instance.

    If parameters are not provided, loads from environment variables.

    Args:
        provider: Provider name (ollama, groq, openai)
        model: Model name
        temperature: Sampling temperature
        api_key: API key
        base_url: Base URL
        tools: Tools to bind

    Returns:
        BaseChatModel instance
    """
    config = LLMConfig.from_env()

    # Override with provided parameters
    if provider:
        config.provider = LLMProvider(provider.lower())
    if model:
        config.model = model
    if temperature is not None:
        config.temperature = temperature
    if api_key:
        config.api_key = api_key
    if base_url:
        config.base_url = base_url

    return LLMFactory.create_from_config(config, tools=tools)
