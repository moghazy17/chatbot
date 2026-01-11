"""
Session Repository.

Manages chat session state and history.
"""

import uuid
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


@dataclass
class Session:
    """Chat session data."""
    session_id: str
    messages: List[BaseMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


class SessionRepository:
    """
    Manages chat sessions.
    
    Stores session state in memory. For production, replace with
    a persistent storage backend (Redis, database, etc.)
    """
    
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
    
    def create_session(self, session_id: Optional[str] = None) -> Session:
        """
        Create a new session.
        
        Args:
            session_id: Optional session ID (generates UUID if not provided)
            
        Returns:
            New Session object
        """
        if not session_id:
            session_id = str(uuid.uuid4())
        
        session = Session(session_id=session_id)
        self._sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get an existing session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session if found, None otherwise
        """
        return self._sessions.get(session_id)
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> Session:
        """
        Get existing session or create new one.
        
        Args:
            session_id: Optional session ID
            
        Returns:
            Session object
        """
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            session.last_activity = datetime.now()
            return session
        
        return self.create_session(session_id)
    
    def add_message(self, session_id: str, message: BaseMessage) -> None:
        """
        Add a message to a session.
        
        Args:
            session_id: Session ID
            message: Message to add
        """
        session = self.get_session(session_id)
        if session:
            session.messages.append(message)
            session.last_activity = datetime.now()
    
    def get_messages(self, session_id: str) -> List[BaseMessage]:
        """
        Get all messages from a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            List of messages
        """
        session = self.get_session(session_id)
        return session.messages if session else []
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear a session's messages.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if session existed
        """
        if session_id in self._sessions:
            self._sessions[session_id].messages = []
            return True
        return False
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session entirely.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if session was deleted
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
    
    def list_sessions(self) -> List[str]:
        """
        List all session IDs.
        
        Returns:
            List of session IDs
        """
        return list(self._sessions.keys())
