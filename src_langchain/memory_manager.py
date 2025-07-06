"""
Memory Manager for LangChain RAG Agents
Implements conversation memory using native LangChain components.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI


class MemoryManager:
    """
    High-level memory manager that can handle multiple conversation sessions.
    Uses LangChain's native ConversationBufferWindowMemory directly.
    """
    
    def __init__(self, default_window_size: int = 5):
        """
        Initialize the memory manager.
        
        Args:
            default_window_size: Default window size for new conversations
        """
        self.default_window_size = default_window_size
        # Use LangChain's memory instances directly
        self.sessions: Dict[str, ConversationBufferWindowMemory] = {}
        # Custom history for additional features (timestamps, metadata)
        self.session_metadata: Dict[str, List[Dict[str, Any]]] = {}
    
    def get_session(self, session_id: str, window_size: Optional[int] = None) -> ConversationBufferWindowMemory:
        """
        Get or create a conversation session.
        
        Args:
            session_id: Unique identifier for the session
            window_size: Optional custom window size for this session
            
        Returns:
            LangChain ConversationBufferWindowMemory instance for the session
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationBufferWindowMemory(k=window_size or self.default_window_size)
            self.session_metadata[session_id] = []
        
        return self.sessions[session_id]
    
    def add_conversation_turn(self, session_id: str, user_message: str, 
                             assistant_response: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a conversation turn to a specific session.
        
        Args:
            session_id: Session identifier
            user_message: The user's input
            assistant_response: The assistant's response
            metadata: Optional metadata for the turn
        """
        session = self.get_session(session_id)
        
        # Add to LangChain memory using proper message types
        session.chat_memory.add_user_message(user_message)
        session.chat_memory.add_ai_message(assistant_response)
        
        # Add to our custom metadata for additional features
        turn_metadata = {
            "user_message": user_message,
            "assistant_response": assistant_response,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.session_metadata[session_id].append(turn_metadata)
        
        # Maintain window size by removing oldest turns from metadata
        window_size = session.k
        if len(self.session_metadata[session_id]) > window_size:
            self.session_metadata[session_id] = self.session_metadata[session_id][-window_size:]
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get the current conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of conversation turns within the window
        """
        return self.session_metadata.get(session_id, []).copy()
    
    def get_langchain_messages(self, session_id: str) -> List[BaseMessage]:
        """
        Get conversation history as LangChain messages for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of LangChain message objects
        """
        session = self.get_session(session_id)
        return session.chat_memory.messages
    
    def get_formatted_history(self, session_id: str, format_type: str = "text") -> str:
        """
        Get conversation history in a formatted string for LLM context.
        
        Args:
            session_id: Session identifier
            format_type: How to format the history ("text", "json", "compact", "langchain")
            
        Returns:
            Formatted conversation history string
        """
        session = self.get_session(session_id)
        
        if format_type == "langchain":
            return session.buffer
        
        metadata = self.session_metadata.get(session_id, [])
        if not metadata:
            return ""
        
        if format_type == "text":
            return self._format_as_text(metadata)
        elif format_type == "json":
            return json.dumps(metadata, indent=2)
        elif format_type == "compact":
            return self._format_as_compact(metadata)
        else:
            raise ValueError(f"Unknown format_type: {format_type}")
    
    def _format_as_text(self, metadata: List[Dict[str, Any]]) -> str:
        """Format conversation history as readable text."""
        formatted_lines = []
        for turn in metadata:
            formatted_lines.append(f"User: {turn['user_message']}")
            formatted_lines.append(f"Assistant: {turn['assistant_response']}")
            formatted_lines.append("")  # Empty line for separation
        
        return "\n".join(formatted_lines).strip()
    
    def _format_as_compact(self, metadata: List[Dict[str, Any]]) -> str:
        """Format conversation history in a compact format for LLM context."""
        formatted_lines = []
        for turn in metadata:
            formatted_lines.append(f"Q: {turn['user_message']}")
            formatted_lines.append(f"A: {turn['assistant_response']}")
        
        return "\n".join(formatted_lines)
    
    def clear_session(self, session_id: str) -> None:
        """
        Clear memory for a specific session.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            self.sessions[session_id].clear()
            self.session_metadata[session_id] = []
    
    def clear_all_sessions(self) -> None:
        """Clear memory for all sessions."""
        for session in self.sessions.values():
            session.clear()
        self.session_metadata.clear()
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all sessions.
        
        Returns:
            Dictionary with session statistics
        """
        stats = {
            "total_sessions": len(self.sessions),
            "sessions": {}
        }
        
        for session_id, session in self.sessions.items():
            metadata = self.session_metadata.get(session_id, [])
            stats["sessions"][session_id] = {
                "total_turns": len(metadata),
                "window_size": session.k,
                "memory_usage_percent": (len(metadata) / session.k) * 100 if session.k > 0 else 0,
                "oldest_turn": metadata[0]["timestamp"] if metadata else None,
                "newest_turn": metadata[-1]["timestamp"] if metadata else None,
                "langchain_messages_count": len(session.chat_memory.messages)
            }
        
        return stats
    
    def get_session_memory(self, session_id: str) -> Optional[ConversationBufferWindowMemory]:
        """
        Get the memory object for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            LangChain ConversationBufferWindowMemory instance for the session, or None if not found
        """
        return self.sessions.get(session_id)
    
    def list_sessions(self) -> List[str]:
        """
        Get a list of all session IDs.
        
        Returns:
            List of session identifiers
        """
        return list(self.sessions.keys())
    
    def remove_session(self, session_id: str) -> bool:
        """
        Remove a session completely.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was removed, False if not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            if session_id in self.session_metadata:
                del self.session_metadata[session_id]
            return True
        return False
    
    def is_memory_full(self, session_id: str) -> bool:
        """
        Check if memory is at capacity for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if memory window is full
        """
        session = self.get_session(session_id)
        return len(self.session_metadata.get(session_id, [])) >= session.k