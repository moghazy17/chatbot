"""
Chatbot Package.

A LangGraph-based chatbot with unified text and voice support.
"""

# Graph module
from .graph import (
    UnifiedChatState,
    create_unified_graph,
    create_chatbot_graph,
    export_mermaid,
    print_graph_structure,
)
from .graph.builder import extract_response

# Tools module
from .tools import ToolsRegistry, tools_registry

# OpenAI client
from .openai_client import get_openai_client

# Speech nodes (legacy - for backward compatibility)
from .nodes import (
    SpeechToTextNode,
    TextToSpeechNode,
    transcribe_audio,
    synthesize_speech,
)

# Realtime client
from .realtime_client import RealtimeClient, RealtimeConfig, create_realtime_client

# Chat modes (handlers)
from .modes import (
    BaseChatHandler,
    ChatMessage,
    ChatMode,
    TextChatHandler,
    VoiceChatHandler,
    RealtimeVoiceHandler,
)

# LLM Provider
from .llm_provider import LLMProvider, LLMFactory, LLMConfig

__all__ = [
    # Graph
    "UnifiedChatState",
    "create_unified_graph",
    "create_chatbot_graph",
    "extract_response",
    "export_mermaid",
    "print_graph_structure",
    # Tools
    "ToolsRegistry",
    "tools_registry",
    # OpenAI client
    "get_openai_client",
    # Speech nodes
    "SpeechToTextNode",
    "TextToSpeechNode",
    "transcribe_audio",
    "synthesize_speech",
    # Realtime client (low-level)
    "RealtimeClient",
    "RealtimeConfig",
    "create_realtime_client",
    # Chat modes (high-level handlers)
    "BaseChatHandler",
    "ChatMessage",
    "ChatMode",
    "TextChatHandler",
    "VoiceChatHandler",
    "RealtimeVoiceHandler",
    # LLM Provider
    "LLMProvider",
    "LLMFactory",
    "LLMConfig",
]
