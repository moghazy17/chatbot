"""
Graph Router.

Handles endpoints for graph visualization and information.
"""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

router = APIRouter(prefix="/api/v1/graph")


@router.get("/", response_class=PlainTextResponse)
async def get_graph_mermaid():
    """
    Get the graph structure as a Mermaid diagram.
    
    Returns:
        Mermaid diagram string
    """
    from chatbot.graph import export_mermaid
    
    return export_mermaid()


@router.get("/info")
async def get_graph_info():
    """
    Get detailed graph information.
    
    Returns:
        Graph structure with nodes and edges
    """
    from chatbot.graph.visualization import get_graph_info
    
    return get_graph_info()


@router.get("/print")
async def print_graph():
    """
    Get a text representation of the graph structure.
    
    Returns:
        Text description of the graph
    """
    lines = []
    lines.append("=" * 60)
    lines.append("UNIFIED CHATBOT GRAPH STRUCTURE")
    lines.append("=" * 60)
    
    lines.append("\nENTRY POINTS:")
    lines.append("  • input_type='text' → prompt")
    lines.append("  • input_type='audio' → stt")
    
    lines.append("\nNODES:")
    nodes = [
        ("stt", "Speech-to-Text", "Transcribes audio to text"),
        ("prompt", "System Prompt", "Adds system prompt if not present"),
        ("llm", "LLM Processor", "Invokes the language model"),
        ("tools", "Tool Executor", "Executes tool calls"),
        ("tts", "Text-to-Speech", "Converts response to audio"),
    ]
    for node_id, name, desc in nodes:
        lines.append(f"  • {node_id}: {name}")
        lines.append(f"    └─ {desc}")
    
    lines.append("\nEDGES:")
    edges = [
        "stt → prompt",
        "prompt → llm",
        "llm → tools (if tool_calls)",
        "llm → output_router (if no tool_calls)",
        "tools → llm (loop back)",
        "output_router → tts (if output_type='audio')",
        "output_router → END (if output_type='text')",
        "tts → END",
    ]
    for edge in edges:
        lines.append(f"  • {edge}")
    
    lines.append("\n" + "=" * 60)
    
    return {"structure": "\n".join(lines)}
