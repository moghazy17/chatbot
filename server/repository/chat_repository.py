"""
Chat Repository.

Handles chat business logic and graph invocation.
"""

from typing import Optional, Tuple, Generator
from langchain_core.messages import HumanMessage, AIMessage

from .session_repository import SessionRepository
from ..config import config


class ChatRepository:
    """
    Repository for chat operations.
    
    Manages graph invocation and session state.
    """
    
    def __init__(
        self,
        provider: str = "ollama",
        model: Optional[str] = None,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.default_provider = provider
        self.default_model = model
        self.default_temperature = temperature
        self.default_api_key = api_key
        self.default_base_url = base_url
        
        self._session_repo = SessionRepository()
        self._graph = None
    
    def _get_graph(self):
        """Get or create the graph."""
        if self._graph is None:
            from chatbot.graph import create_unified_graph
            self._graph = create_unified_graph()
        return self._graph
    
    def _build_config(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """Build LLM config dict."""
        # Determine which provider to use
        effective_provider = provider or self.default_provider
        
        # Get the correct API key for the provider
        api_key = config.get_api_key(effective_provider)
        if not api_key:
            api_key = self.default_api_key
        
        return {
            "provider": effective_provider,
            "model": model,  # Let the LLM node handle None -> default
            "temperature": temperature if temperature is not None else self.default_temperature,
            "api_key": api_key,
            "base_url": self.default_base_url,
        }
    
    def send_text(
        self,
        message: str,
        session_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, str]:
        """
        Send a text message and get response.
        
        Args:
            message: User message
            session_id: Optional session ID
            provider: Optional LLM provider override
            model: Optional model override
            temperature: Optional temperature override
            
        Returns:
            Tuple of (response_text, session_id)
        """
        # Get or create session
        session = self._session_repo.get_or_create_session(session_id)
        
        # Add user message to session
        self._session_repo.add_message(session.session_id, HumanMessage(content=message))
        
        # Build state
        graph = self._get_graph()
        state = {
            "messages": session.messages,
            "input_type": "text",
            "output_type": "text",
            "audio_input": None,
            "audio_output": None,
            "transcription": None,
            "llm_config": self._build_config(provider, model, temperature),
        }
        
        # Invoke graph
        result = graph.invoke(state)
        
        # Extract response
        response_text = ""
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                response_text = msg.content
                break
        
        # Update session with new messages
        session.messages = list(result.get("messages", []))
        
        return response_text, session.session_id
    
    def send_audio(
        self,
        audio_bytes: bytes,
        session_id: Optional[str] = None,
        generate_audio: bool = True,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Tuple[str, str, str, Optional[bytes]]:
        """
        Send audio and get response.
        
        Args:
            audio_bytes: Audio data
            session_id: Optional session ID
            generate_audio: Whether to generate audio response
            provider: Optional LLM provider override
            model: Optional model override
            
        Returns:
            Tuple of (response_text, transcription, session_id, audio_output)
        """
        # Get or create session
        session = self._session_repo.get_or_create_session(session_id)
        
        # Build state
        graph = self._get_graph()
        state = {
            "messages": session.messages,
            "input_type": "audio",
            "output_type": "audio" if generate_audio else "text",
            "audio_input": audio_bytes,
            "audio_output": None,
            "transcription": None,
            "llm_config": self._build_config(provider, model),
        }
        
        # Invoke graph
        result = graph.invoke(state)
        
        # Extract response
        response_text = ""
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                response_text = msg.content
                break
        
        transcription = result.get("transcription", "")
        audio_output = result.get("audio_output")
        
        # Update session with new messages
        session.messages = list(result.get("messages", []))
        
        return response_text, transcription, session.session_id, audio_output
    
    def stream_text(
        self,
        message: str,
        session_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """
        Stream a text response.
        
        Args:
            message: User message
            session_id: Optional session ID
            provider: Optional LLM provider override
            model: Optional model override
            temperature: Optional temperature override
            
        Yields:
            Response chunks
        """
        # Get or create session
        session = self._session_repo.get_or_create_session(session_id)
        
        # Add user message
        self._session_repo.add_message(session.session_id, HumanMessage(content=message))
        
        # Build state
        graph = self._get_graph()
        state = {
            "messages": session.messages,
            "input_type": "text",
            "output_type": "text",
            "audio_input": None,
            "audio_output": None,
            "transcription": None,
            "llm_config": self._build_config(provider, model, temperature),
        }
        
        # Stream from graph
        collected_response = ""
        for chunk in graph.stream(state):
            for node_name, node_output in chunk.items():
                if node_name == "llm":
                    messages = node_output.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, AIMessage) and msg.content:
                            collected_response = msg.content
                            yield msg.content
        
        # Update session
        if collected_response:
            self._session_repo.add_message(
                session.session_id,
                AIMessage(content=collected_response)
            )
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a session."""
        return self._session_repo.clear_session(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        return self._session_repo.delete_session(session_id)
