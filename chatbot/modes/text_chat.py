"""
Text Chat Handler.

Handles standard text-based chat interactions using LangGraph.
"""

from typing import Optional, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from .base import BaseChatHandler, ChatMessage, ChatMode
from ..graph import create_unified_graph


class TextChatHandler(BaseChatHandler):
    """
    Text-based chat handler.
    
    Uses LangGraph for processing messages with optional tool calling.
    """

    @property
    def mode(self) -> ChatMode:
        return ChatMode.TEXT

    def initialize(self) -> bool:
        """Create and compile the chat graph."""
        try:
            self._graph = create_unified_graph()
            self._llm_config = {
                "provider": self.provider,
                "model": self.model,
                "temperature": self.temperature,
                "api_key": self.api_key,
                "base_url": self.base_url,
            }
            self._initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize TextChatHandler: {e}")
            self._initialized = False
            return False

    def send_message(self, content: str) -> ChatMessage:
        """
        Send a text message and get the AI response.
        
        Args:
            content: User's message text
            
        Returns:
            ChatMessage with assistant's response
        """
        if not self._initialized:
            raise RuntimeError("Handler not initialized. Call initialize() first.")

        # Add user message to history
        self.add_message("user", content)

        # Build LangChain messages for the graph
        lc_messages = self.langchain_messages

        # Invoke the graph with unified state
        result = self._graph.invoke({
            "messages": lc_messages,
            "input_type": "text",
            "output_type": "text",
            "audio_input": None,
            "audio_output": None,
            "transcription": None,
            "llm_config": self._llm_config,
        })

        # Extract the response
        response_messages = result.get("messages", [])
        
        # Find the last AI message with content
        assistant_response = ""
        for msg in reversed(response_messages):
            if isinstance(msg, AIMessage) and msg.content:
                assistant_response = msg.content
                break

        # Add response to history
        response_msg = self.add_message("assistant", assistant_response)
        
        return response_msg

    def get_streaming_response(self, content: str):
        """
        Send a message and stream the response.
        
        Args:
            content: User's message text
            
        Yields:
            Response chunks as they arrive
        """
        if not self._initialized:
            raise RuntimeError("Handler not initialized. Call initialize() first.")

        # Add user message
        self.add_message("user", content)
        lc_messages = self.langchain_messages

        # Stream from the graph with unified state
        collected_response = ""
        state = {
            "messages": lc_messages,
            "input_type": "text",
            "output_type": "text",
            "audio_input": None,
            "audio_output": None,
            "transcription": None,
            "llm_config": self._llm_config,
        }
        for chunk in self._graph.stream(state):
            for node_name, node_output in chunk.items():
                if node_name == "llm":
                    messages = node_output.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, AIMessage) and msg.content:
                            collected_response = msg.content
                            yield msg.content

        # Add final response to history
        if collected_response:
            self.add_message("assistant", collected_response)
