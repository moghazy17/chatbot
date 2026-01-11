"""
LLM Node.

Invokes the LLM with the current messages and returns the response.
"""

from typing import Literal, Optional, Any
from langchain_core.messages import AIMessage
from langgraph.graph import END

from ..state import UnifiedChatState
from ...llm_provider import LLMProvider, LLMFactory


# Cache for LLM instances
_llm_cache: dict[str, Any] = {}


def _get_llm(config: Optional[dict], tools: Optional[list] = None):
    """Get or create an LLM instance based on config."""
    if not config:
        config = {}
    
    # Get provider
    provider_str = config.get("provider") or "ollama"
    try:
        provider = LLMProvider(provider_str.lower())
    except ValueError:
        provider = LLMProvider.OLLAMA
    
    # Get model - use provider's default if not specified or None
    model = config.get("model")
    if not model:
        default_models = LLMFactory.get_default_models(provider)
        model = default_models[0] if default_models else "llama3.2"
    
    # Get temperature
    temperature = config.get("temperature")
    if temperature is None:
        temperature = 0.7
    
    # Create cache key
    cache_key = f"{provider}:{model}:{temperature}"
    
    if cache_key not in _llm_cache:
        _llm_cache[cache_key] = LLMFactory.create(
            provider=provider,
            model=model,
            temperature=temperature,
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            tools=tools,
        )
    
    return _llm_cache[cache_key]


def llm_node(state: UnifiedChatState) -> dict:
    """
    Invoke the LLM with current messages.
    
    Args:
        state: Current graph state with messages
        
    Returns:
        Updated state with LLM response added to messages
    """
    messages = list(state.get("messages", []))
    llm_config = state.get("llm_config")
    
    # Import tools here to avoid circular imports
    from ...tools import tools_registry
    tools = tools_registry.get_tools()
    
    # Get LLM instance
    llm = _get_llm(llm_config, tools if tools else None)
    
    # Invoke LLM
    response = llm.invoke(messages)
    
    return {"messages": [response]}


def should_use_tools(state: UnifiedChatState) -> Literal["tools", "output"]:
    """
    Determine if we should route to tools or to output.
    
    Args:
        state: Current graph state
        
    Returns:
        "tools" if LLM wants to call tools, "output" otherwise
    """
    messages = state.get("messages", [])
    
    if not messages:
        return "output"
    
    last_message = messages[-1]
    
    # Check if the AI wants to call tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"\n{'='*80}")
        print(f"🤖 LLM DECISION: Calling {len(last_message.tool_calls)} tool(s)")
        for tool_call in last_message.tool_calls:
            print(f"   → Tool: {tool_call['name']}")
            print(f"   → Args: {tool_call.get('args', {})}")
        print(f"{'='*80}\n")
        return "tools"
    
    return "output"
