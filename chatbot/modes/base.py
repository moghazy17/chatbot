"""
Base Chat Handler.

Abstract base class defining the interface for all chat modes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class ChatMode(Enum):
    """Available chat interaction modes."""
    TEXT = "text"
    VOICE = "voice"
    REALTIME = "realtime"


@dataclass
class ChatMessage:
    """Unified message format for all chat modes."""
    role: str  # "user" or "assistant"
    content: str
    audio: Optional[bytes] = None
    metadata: dict = field(default_factory=dict)

    def to_langchain(self) -> BaseMessage:
        """Convert to LangChain message format."""
        if self.role == "user":
            return HumanMessage(content=self.content)
        return AIMessage(content=self.content)

    @classmethod
    def from_langchain(cls, msg: BaseMessage) -> "ChatMessage":
        """Create from LangChain message."""
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        return cls(role=role, content=msg.content)


class BaseChatHandler(ABC):
    """
    Abstract base class for chat handlers.
    
    Each handler manages its own conversation state and interaction logic.
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: Optional[str] = None,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the chat handler.
        
        Args:
            provider: LLM provider ("ollama", "groq", "openai")
            model: Model name/ID
            temperature: Sampling temperature
            api_key: API key for cloud providers
            base_url: Custom base URL
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.base_url = base_url
        
        self._messages: List[ChatMessage] = []
        self._graph = None
        self._initialized = False

    @property
    def mode(self) -> ChatMode:
        """Return the chat mode this handler implements."""
        raise NotImplementedError

    @property
    def messages(self) -> List[ChatMessage]:
        """Get conversation history."""
        return self._messages

    @property
    def langchain_messages(self) -> List[BaseMessage]:
        """Get conversation history as LangChain messages."""
        return [msg.to_langchain() for msg in self._messages]

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._messages = []

    def add_message(self, role: str, content: str, audio: Optional[bytes] = None) -> ChatMessage:
        """Add a message to history."""
        msg = ChatMessage(role=role, content=content, audio=audio)
        self._messages.append(msg)
        return msg

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the handler (create graph, connections, etc.)
        
        Returns:
            True if initialization successful.
        """
        pass

    @abstractmethod
    def send_message(self, content: str) -> ChatMessage:
        """
        Send a text message and get response.
        
        Args:
            content: Message text
            
        Returns:
            Assistant's response message
        """
        pass

    def is_ready(self) -> bool:
        """Check if handler is ready to process messages."""
        return self._initialized
