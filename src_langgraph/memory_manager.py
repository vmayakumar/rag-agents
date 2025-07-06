"""
Memory Manager for LangGraph RAG Agent
Uses LangGraph state management for conversation memory.
"""

from typing import List, Dict, Any, Optional, TypedDict
from datetime import datetime
import json
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage


class RAGState(TypedDict):
    """State definition for LangGraph RAG workflow."""
    user_query: str
    retrieved_chunks: List[Dict[str, Any]]
    generated_answer: str
    conversation_history: List[Dict[str, Any]]
    confidence: float
    needs_clarification: bool
    error_count: int
    session_id: str


class LangGraphMemoryManager:
    """
    Memory manager that works with LangGraph state.
    Integrates LangChain memory components with LangGraph state management.
    """
    
    def __init__(self, default_window_size: int = 5):
        """
        Initialize the LangGraph memory manager.
        
        Args:
            default_window_size: Default window size for conversation memory
        """
        self.default_window_size = default_window_size
        self.sessions: Dict[str, ConversationBufferWindowMemory] = {}
    
    def get_session(self, session_id: str, window_size: Optional[int] = None) -> ConversationBufferWindowMemory:
        """
        Get or create a LangChain memory session.
        
        Args:
            session_id: Unique identifier for the session
            window_size: Optional custom window size for this session
            
        Returns:
            LangChain ConversationBufferWindowMemory instance
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationBufferWindowMemory(
                k=window_size or self.default_window_size
            )
        
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
        
        # Add to LangChain memory
        session.chat_memory.add_user_message(user_message)
        session.chat_memory.add_ai_message(assistant_response)
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session in a format suitable for LangGraph state.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of conversation turns
        """
        session = self.get_session(session_id)
        messages = session.chat_memory.messages
        
        history = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                history.append({
                    "user_message": messages[i].content,
                    "assistant_response": messages[i + 1].content,
                    "timestamp": datetime.now().isoformat()
                })
        
        return history
    
    def get_formatted_history(self, session_id: str, format_type: str = "compact") -> str:
        """
        Get conversation history in a formatted string.
        
        Args:
            session_id: Session identifier
            format_type: How to format the history ("text", "json", "compact", "langchain")
            
        Returns:
            Formatted conversation history string
        """
        session = self.get_session(session_id)
        
        if format_type == "langchain":
            return session.buffer
        
        history = self.get_conversation_history(session_id)
        if not history:
            return ""
        
        if format_type == "text":
            return self._format_as_text(history)
        elif format_type == "json":
            return json.dumps(history, indent=2)
        elif format_type == "compact":
            return self._format_as_compact(history)
        else:
            raise ValueError(f"Unknown format_type: {format_type}")
    
    def _format_as_text(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history as readable text."""
        formatted_lines = []
        for turn in history:
            formatted_lines.append(f"User: {turn['user_message']}")
            formatted_lines.append(f"Assistant: {turn['assistant_response']}")
            formatted_lines.append("")  # Empty line for separation
        
        return "\n".join(formatted_lines).strip()
    
    def _format_as_compact(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history in a compact format for LLM context."""
        formatted_lines = []
        for turn in history:
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
    
    def clear_all_sessions(self) -> None:
        """Clear memory for all sessions."""
        for session in self.sessions.values():
            session.clear()
    
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
            history = self.get_conversation_history(session_id)
            stats["sessions"][session_id] = {
                "total_turns": len(history),
                "window_size": session.k,
                "memory_usage_percent": (len(history) / session.k) * 100 if session.k > 0 else 0,
                "oldest_turn": history[0]["timestamp"] if history else None,
                "newest_turn": history[-1]["timestamp"] if history else None,
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
            return True
        return False 