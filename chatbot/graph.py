"""
LangGraph Chatbot Graph.

This module defines the chatbot's conversation graph using LangGraph.
The graph supports tool calling when tools are registered.
"""

from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .tools import tools_registry


class ChatState(TypedDict):
    """State for the chatbot graph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


def create_chatbot_graph(
    model_name: str = "llama3.2",
    temperature: float = 0.7,
    base_url: str = "http://localhost:11434",
):
    """
    Create and compile the chatbot graph.
    
    Args:
        model_name: The Ollama model to use.
        temperature: Sampling temperature for the model.
        base_url: The Ollama server URL.
    
    Returns:
        A compiled LangGraph that can process messages.
    """
    
    # Get tools from registry
    tools = tools_registry.get_tools()
    
    # Initialize the LLM
    llm = ChatOllama(model=model_name, temperature=temperature, base_url=base_url)
    
    # Bind tools if any are registered
    if tools:
        llm = llm.bind_tools(tools)
    
    def chatbot_node(state: ChatState) -> dict:
        """Process messages and generate a response."""
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    
    def should_use_tools(state: ChatState) -> str:
        """Determine if we should route to tools or end."""
        last_message = state["messages"][-1]
        
        # Check if the AI wants to call tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
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
