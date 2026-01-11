"""
Tools Registry.

Provides a centralized registry for managing LLM tools.
Tools can be registered via decorator or direct method call.
"""

from typing import Callable, List, Optional
from langchain_core.tools import BaseTool, tool


class ToolsRegistry:
    """
    A registry to manage tools for the chatbot.
    
    This allows you to easily add, remove, and list tools
    that the chatbot can use.
    """
    
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
    
    def register_tool(self, tool_func: BaseTool) -> BaseTool:
        """Register a tool in the registry."""
        self._tools[tool_func.name] = tool_func
        return tool_func
    
    def register(self, func: Callable) -> BaseTool:
        """
        Decorator to register a function as a tool.
        
        Usage:
            @tools_registry.register
            def my_tool(arg: str) -> str:
                '''Tool description.'''
                return result
        """
        wrapped = tool(func)
        self.register_tool(wrapped)
        return wrapped
    
    def unregister(self, tool_name: str) -> Optional[BaseTool]:
        """Remove a tool from the registry."""
        return self._tools.pop(tool_name, None)
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a specific tool by name."""
        return self._tools.get(tool_name)
    
    def get_tools(self) -> List[BaseTool]:
        """Get all registered tools as a list."""
        return list(self._tools.values())
    
    def list_tools(self) -> List[str]:
        """List all tool names."""
        return list(self._tools.keys())
    
    def has_tools(self) -> bool:
        """Check if any tools are registered."""
        return len(self._tools) > 0
    
    def clear(self) -> None:
        """Remove all tools from the registry."""
        self._tools.clear()
    
    def get_tools_schema(self) -> List[dict]:
        """Get schema for all registered tools."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.args_schema.schema() if t.args_schema else {},
            }
            for t in self._tools.values()
        ]


# Global tools registry instance
tools_registry = ToolsRegistry()
