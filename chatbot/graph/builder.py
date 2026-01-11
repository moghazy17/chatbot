"""
Graph Builder Module.

Constructs the unified LangGraph that handles both text and voice inputs.
"""

from typing import Optional, Any
from langgraph.graph import StateGraph, END

from .state import UnifiedChatState
from .nodes import (
    route_by_input_type,
    stt_node,
    prompt_node,
    llm_node,
    should_use_tools,
    tools_node,
    tts_node,
    should_generate_audio,
)


def create_unified_graph(
    llm_config: Optional[dict[str, Any]] = None,
):
    """
    Create the unified chatbot graph.
    
    This graph handles both text and audio inputs with conditional routing:
    - Text input → Prompt → LLM → [Tools] → Output
    - Audio input → STT → Prompt → LLM → [Tools] → [TTS] → Output
    
    Args:
        llm_config: Optional LLM configuration dict with keys:
            - provider: "ollama", "groq", "openai"
            - model: Model name
            - temperature: Sampling temperature
            - api_key: API key (for cloud providers)
            - base_url: Custom base URL
            - tts_voice: Voice for TTS output
    
    Returns:
        Compiled LangGraph
    """
    # Build the graph
    graph_builder = StateGraph(UnifiedChatState)
    
    # Add nodes
    graph_builder.add_node("stt", stt_node)
    graph_builder.add_node("prompt", prompt_node)
    graph_builder.add_node("llm", llm_node)
    graph_builder.add_node("tools", tools_node)
    graph_builder.add_node("tts", tts_node)
    
    # Set entry point with conditional routing
    graph_builder.set_conditional_entry_point(
        route_by_input_type,
        {
            "stt": "stt",
            "prompt": "prompt",
        }
    )
    
    # STT → Prompt
    graph_builder.add_edge("stt", "prompt")
    
    # Prompt → LLM
    graph_builder.add_edge("prompt", "llm")
    
    # LLM → Tools or Output
    graph_builder.add_conditional_edges(
        "llm",
        should_use_tools,
        {
            "tools": "tools",
            "output": "output_router",
        }
    )
    
    # Tools → LLM (loop back)
    graph_builder.add_edge("tools", "llm")
    
    # Add output routing node
    def output_router(state: UnifiedChatState) -> dict:
        """Pass-through node for output routing."""
        return {}
    
    graph_builder.add_node("output_router", output_router)
    
    # Output → TTS or END
    graph_builder.add_conditional_edges(
        "output_router",
        should_generate_audio,
        {
            "tts": "tts",
            "end": END,
        }
    )
    
    # TTS → END
    graph_builder.add_edge("tts", END)
    
    # Compile
    return graph_builder.compile()


def create_chatbot_graph(
    provider: str = "ollama",
    model: Optional[str] = None,
    temperature: float = 0.7,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """
    Create a chatbot graph with the specified LLM configuration.
    
    This is a convenience wrapper for backward compatibility.
    
    Args:
        provider: LLM provider name ("ollama", "groq", "openai")
        model: Model name/ID (uses provider default if not specified)
        temperature: Sampling temperature for the model
        api_key: API key (required for groq/openai)
        base_url: Base URL (for ollama or custom endpoints)
    
    Returns:
        Compiled LangGraph
    """
    llm_config = {
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "api_key": api_key,
        "base_url": base_url,
    }
    
    return create_unified_graph(llm_config=llm_config)


def extract_response(result: dict) -> str:
    """
    Extract the assistant response from graph result.
    
    Args:
        result: Graph invocation result
        
    Returns:
        Assistant's text response
    """
    from langchain_core.messages import AIMessage
    
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return ""
