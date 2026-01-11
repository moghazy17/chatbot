"""
Graph Visualization Module.

Provides utilities to visualize and export the graph structure.
"""

from typing import Optional


def export_mermaid(graph=None) -> str:
    """
    Export the graph structure as a Mermaid diagram.
    
    Args:
        graph: Optional compiled graph. If None, generates the standard flow.
        
    Returns:
        Mermaid diagram string
    """
    # Generate the standard unified graph flow
    mermaid = """flowchart TD
    Start([Input]) --> Router{input_type?}
    
    Router -->|audio| STT[STT Node]
    Router -->|text| Prompt[System Prompt Node]
    
    STT --> Prompt
    Prompt --> LLM[LLM Node]
    
    LLM --> ToolCheck{has_tool_calls?}
    ToolCheck -->|yes| Tools[Tool Executor]
    Tools --> LLM
    
    ToolCheck -->|no| OutputCheck{output_type?}
    OutputCheck -->|audio| TTS[TTS Node]
    OutputCheck -->|text| EndText([Text Response])
    
    TTS --> EndAudio([Audio Response])"""
    
    return mermaid


def print_graph_structure(graph=None) -> None:
    """
    Print a summary of the graph structure to console.
    
    Args:
        graph: Optional compiled graph
    """
    print("\n" + "=" * 60)
    print("UNIFIED CHATBOT GRAPH STRUCTURE")
    print("=" * 60)
    
    print("\n📥 ENTRY POINTS:")
    print("   • input_type='text' → prompt")
    print("   • input_type='audio' → stt")
    
    print("\n🔗 NODES:")
    nodes = [
        ("stt", "Speech-to-Text", "Transcribes audio to text"),
        ("prompt", "System Prompt", "Adds system prompt if not present"),
        ("llm", "LLM Processor", "Invokes the language model"),
        ("tools", "Tool Executor", "Executes tool calls"),
        ("tts", "Text-to-Speech", "Converts response to audio"),
    ]
    for node_id, name, desc in nodes:
        print(f"   • {node_id}: {name}")
        print(f"     └─ {desc}")
    
    print("\n🔀 EDGES:")
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
        print(f"   • {edge}")
    
    print("\n" + "=" * 60)


def get_graph_info() -> dict:
    """
    Get graph information as a dictionary.
    
    Returns:
        Dict with nodes, edges, and metadata
    """
    return {
        "nodes": [
            {"id": "stt", "name": "Speech-to-Text", "type": "processor"},
            {"id": "prompt", "name": "System Prompt", "type": "processor"},
            {"id": "llm", "name": "LLM Processor", "type": "processor"},
            {"id": "tools", "name": "Tool Executor", "type": "processor"},
            {"id": "output_router", "name": "Output Router", "type": "router"},
            {"id": "tts", "name": "Text-to-Speech", "type": "processor"},
        ],
        "edges": [
            {"from": "START", "to": "stt", "condition": "input_type='audio'"},
            {"from": "START", "to": "prompt", "condition": "input_type='text'"},
            {"from": "stt", "to": "prompt"},
            {"from": "prompt", "to": "llm"},
            {"from": "llm", "to": "tools", "condition": "has_tool_calls"},
            {"from": "llm", "to": "output_router", "condition": "no_tool_calls"},
            {"from": "tools", "to": "llm"},
            {"from": "output_router", "to": "tts", "condition": "output_type='audio'"},
            {"from": "output_router", "to": "END", "condition": "output_type='text'"},
            {"from": "tts", "to": "END"},
        ],
        "entry_conditions": {
            "audio": "stt",
            "text": "prompt",
        },
    }
