"""
Tools Router.

Handles endpoints for listing and managing tools.
"""

from typing import List
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/tools")


class ToolInfo(BaseModel):
    """Tool information model."""
    name: str
    description: str
    parameters: dict


@router.get("/", response_model=List[ToolInfo])
async def list_tools():
    """
    List all registered tools.
    
    Returns:
        List of tool information
    """
    from chatbot.tools import tools_registry
    
    tools = []
    for tool in tools_registry.get_tools():
        tools.append(ToolInfo(
            name=tool.name,
            description=tool.description or "",
            parameters=tool.args_schema.schema() if tool.args_schema else {},
        ))
    
    return tools


@router.get("/names")
async def list_tool_names():
    """
    List all registered tool names.
    
    Returns:
        List of tool names
    """
    from chatbot.tools import tools_registry
    
    return {"tools": tools_registry.list_tools()}


@router.get("/{tool_name}")
async def get_tool(tool_name: str):
    """
    Get information about a specific tool.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Tool information
    """
    from chatbot.tools import tools_registry
    
    tool = tools_registry.get_tool(tool_name)
    if not tool:
        return {"error": f"Tool '{tool_name}' not found"}
    
    return ToolInfo(
        name=tool.name,
        description=tool.description or "",
        parameters=tool.args_schema.schema() if tool.args_schema else {},
    )
