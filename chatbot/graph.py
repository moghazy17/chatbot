"""
LangGraph Chatbot Graph.

This module defines the chatbot's conversation graph using LangGraph.
The graph supports tool calling when tools are registered.
Supports multiple LLM providers: Ollama, Groq, OpenAI.
"""

from typing import Annotated, TypedDict, Sequence, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .tools import tools_registry
from .llm_provider import LLMProvider, LLMFactory, LLMConfig


# System prompt to guide tool usage
SYSTEM_PROMPT = """You are a helpful assistant. You have access to tools that can help you answer questions.

IMPORTANT RULES FOR TOOL USAGE:
1. Only use tools when the user's request REQUIRES information you don't have or an action you can't perform directly.
2. For simple greetings (hi, hello, how are you), casual conversation, or questions you can answer from your knowledge - DO NOT use tools. Just respond naturally.
3. Use tools ONLY when:
   - The user asks about specific hotel reservations, bookings, or guest information
   - The user needs real-time data you cannot provide from memory
   - The user explicitly asks you to perform an action (create, update, delete something)

If you're unsure whether to use a tool, prefer responding directly first."""


class ChatState(TypedDict):
    """State for the chatbot graph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


def create_chatbot_graph(
    provider: str = "ollama",
    model: Optional[str] = None,
    temperature: float = 0.7,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """
    Create and compile the chatbot graph.

    Args:
        provider: LLM provider name ("ollama", "groq", "openai")
        model: Model name/ID (uses provider default if not specified)
        temperature: Sampling temperature for the model
        api_key: API key (required for groq/openai)
        base_url: Base URL (for ollama or custom endpoints)

    Returns:
        A compiled LangGraph that can process messages.
    """
    # Get tools from registry
    tools = tools_registry.get_tools()

    # Convert provider string to enum
    try:
        provider_enum = LLMProvider(provider.lower())
    except ValueError:
        print(f"Warning: Unknown provider '{provider}', defaulting to ollama")
        provider_enum = LLMProvider.OLLAMA

    # Set default model if not specified
    if not model:
        default_models = LLMFactory.get_default_models(provider_enum)
        model = default_models[0] if default_models else "llama3.2"

    # Create LLM instance with tools bound
    llm = LLMFactory.create(
        provider=provider_enum,
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        tools=tools if tools else None,
    )

    def chatbot_node(state: ChatState) -> dict:
        """Process messages and generate a response."""
        messages = list(state["messages"])

        # Add system prompt if not already present and tools are available
        if tools and (not messages or not isinstance(messages[0], SystemMessage)):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        response = llm.invoke(messages)
        return {"messages": [response]}

    def should_use_tools(state: ChatState) -> str:
        """Determine if we should route to tools or end."""
        last_message = state["messages"][-1]

        # Check if the AI wants to call tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print(f"\n{'='*80}")
            print(f"🤖 LLM DECISION: Calling {len(last_message.tool_calls)} tool(s)")
            for tool_call in last_message.tool_calls:
                print(f"   → Tool: {tool_call['name']}")
                print(f"   → Args: {tool_call.get('args', {})}")
            print(f"{'='*80}\n")
            return "tools"
        return END

    # Build the graph
    graph_builder = StateGraph(ChatState)

    # Add the chatbot node
    graph_builder.add_node("chatbot", chatbot_node)

    # Add tool node if tools are registered
    if tools:
        tool_node = ToolNode(tools)
        graph_builder.add_node("tools", tool_node)

        # Add conditional edges for tool routing
        graph_builder.add_conditional_edges(
            "chatbot",
            should_use_tools,
            {
                "tools": "tools",
                END: END,
            }
        )

        # Tools always return to chatbot
        graph_builder.add_edge("tools", "chatbot")
    else:
        # No tools - chatbot goes directly to end
        graph_builder.add_edge("chatbot", END)

    # Set the entry point
    graph_builder.set_entry_point("chatbot")

    # Compile and return the graph
    return graph_builder.compile()


def chat(graph, messages: list, user_input: str) -> tuple[list, str]:
    """
    Process a user message and return the updated messages and response.

    Args:
        graph: The compiled chatbot graph.
        messages: Current conversation history.
        user_input: The user's new message.

    Returns:
        Tuple of (updated_messages, assistant_response)
    """
    # Add user message
    messages.append(HumanMessage(content=user_input))

    # Run the graph
    result = graph.invoke({"messages": messages})

    # Extract the final response
    updated_messages = list(result["messages"])

    # Get the last AI message (skip tool messages)
    assistant_response = ""
    for msg in reversed(updated_messages):
        if isinstance(msg, AIMessage) and msg.content:
            assistant_response = msg.content
            break

    return updated_messages, assistant_response
