"""
Text Chat Handler.

Handles standard text-based chat interactions using LangGraph.
"""

from typing import Optional, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from .base import BaseChatHandler, ChatMessage, ChatMode
from ..graph import create_chatbot_graph


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
            self._graph = create_chatbot_graph(
                provider=self.provider,
                model=self.model,
                temperature=self.temperature,
                api_key=self.api_key,
                base_url=self.base_url,
            )
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

        # Invoke the graph
        result = self._graph.invoke({"messages": lc_messages})

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

        # Stream from the graph
        collected_response = ""
        for chunk in self._graph.stream({"messages": lc_messages}):
            for node_name, node_output in chunk.items():
                if node_name == "chatbot":
                    messages = node_output.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, AIMessage) and msg.content:
                            collected_response = msg.content
                            yield msg.content

        # Add final response to history
        if collected_response:
            self.add_message("assistant", collected_response)
