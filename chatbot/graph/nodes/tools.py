"""
Tool Executor Node.

Executes tool calls from the LLM and returns the results.
"""

from langgraph.prebuilt import ToolNode
from ..state import UnifiedChatState


# Lazy-loaded tool node
_tool_node = None


def _get_tool_node():
    """Get or create the tool node."""
    global _tool_node
    
    if _tool_node is None:
        from ...tools import tools_registry
        tools = tools_registry.get_tools()
        if tools:
            _tool_node = ToolNode(tools)
    
    return _tool_node


def tools_node(state: UnifiedChatState) -> dict:
    """
    Execute tool calls from the LLM.
    
    Args:
        state: Current graph state with tool calls in last message
        
    Returns:
        Updated state with tool results added to messages
    """
    tool_node = _get_tool_node()
    
    if tool_node is None:
        print("⚠️ No tools registered, skipping tool execution")
        return {}
    
    # ToolNode expects a dict with messages
    result = tool_node.invoke({"messages": state.get("messages", [])})
    
    return result
