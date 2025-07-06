"""
Memory Manager for RAG Agents
Implements conversation memory with configurable window size.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import json


class ConversationBufferWindowMemory:
    """
    Memory system that maintains a sliding window of conversation history.
    Keeps only the most recent N conversation turns to prevent context overflow.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize the conversation memory with a sliding window.
        
        Args:
            window_size: Number of conversation turns to keep in memory
        """
        self.window_size = window_size
        self.conversation_history: List[Dict[str, Any]] = []
    
    def add_conversation_turn(self, user_message: str, assistant_response: str, 
                             metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a new conversation turn to memory.
        
        Args:
            user_message: The user's input
            assistant_response: The assistant's response
            metadata: Optional metadata (e.g., timestamp, source, etc.)
        """
        turn = {
            "user_message": user_message,
            "assistant_response": assistant_response,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.conversation_history.append(turn)
        
        # Maintain window size by removing oldest turns
        if len(self.conversation_history) > self.window_size:
            self.conversation_history = self.conversation_history[-self.window_size:]
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the current conversation history.
        
        Returns:
            List of conversation turns within the window
        """
        return self.conversation_history.copy()
    
    def get_formatted_history(self, format_type: str = "text") -> str:
        """
        Get conversation history in a formatted string for LLM context.
        
        Args:
            format_type: How to format the history ("text", "json", "compact")
            
        Returns:
            Formatted conversation history string
        """
        if not self.conversation_history:
            return ""
        
        if format_type == "text":
            return self._format_as_text()
        elif format_type == "json":
            return json.dumps(self.conversation_history, indent=2)
        elif format_type == "compact":
            return self._format_as_compact()
        else:
            raise ValueError(f"Unknown format_type: {format_type}")
    
    def _format_as_text(self) -> str:
        """Format conversation history as readable text."""
        formatted_lines = []
        for turn in self.conversation_history:
            formatted_lines.append(f"User: {turn['user_message']}")
            formatted_lines.append(f"Assistant: {turn['assistant_response']}")
            formatted_lines.append("")  # Empty line for separation
        
        return "\n".join(formatted_lines).strip()
    
    def _format_as_compact(self) -> str:
        """Format conversation history in a compact format for LLM context."""
        formatted_lines = []
        for turn in self.conversation_history:
            formatted_lines.append(f"Q: {turn['user_message']}")
            formatted_lines.append(f"A: {turn['assistant_response']}")
        
        return "\n".join(formatted_lines)
    
    def clear_memory(self) -> None:
        """Clear all conversation history."""
        self.conversation_history = []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current memory state.
        
        Returns:
            Dictionary with memory statistics
        """
        return {
            "total_turns": len(self.conversation_history),
            "window_size": self.window_size,
            "memory_usage_percent": (len(self.conversation_history) / self.window_size) * 100,
            "oldest_turn": self.conversation_history[0]["timestamp"] if self.conversation_history else None,
            "newest_turn": self.conversation_history[-1]["timestamp"] if self.conversation_history else None
        }
    
    def is_memory_full(self) -> bool:
        """
        Check if memory is at capacity.
        
        Returns:
            True if memory window is full
        """
        return len(self.conversation_history) >= self.window_size


class MemoryManager:
    """
    High-level memory manager that can handle multiple conversation sessions.
    Useful for applications that need to manage memory for different users or sessions.
    """
    
    def __init__(self, default_window_size: int = 5):
        """
        Initialize the memory manager.
        
        Args:
            default_window_size: Default window size for new conversations
        """
        self.default_window_size = default_window_size
        self.sessions: Dict[str, ConversationBufferWindowMemory] = {}
    
    def get_session(self, session_id: str, window_size: Optional[int] = None) -> ConversationBufferWindowMemory:
        """
        Get or create a conversation session.
        
        Args:
            session_id: Unique identifier for the session
            window_size: Optional custom window size for this session
            
        Returns:
            Conversation memory for the session
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationBufferWindowMemory(
                window_size or self.default_window_size
            )
        
        return self.sessions[session_id]
    
    def add_conversation_turn(self, session_id: str, user_message: str, 
                             assistant_response: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a conversation turn to a specific session.
        
        Args:
            session_id: Session identifier
            user_message: User's message
            assistant_response: Assistant's response
            metadata: Optional metadata
        """
        session = self.get_session(session_id)
        session.add_conversation_turn(user_message, assistant_response, metadata)
    
    def clear_session(self, session_id: str) -> None:
        """
        Clear memory for a specific session.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            self.sessions[session_id].clear_memory()
    
    def clear_all_sessions(self) -> None:
        """Clear memory for all sessions."""
        for session in self.sessions.values():
            session.clear_memory()
    
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
            stats["sessions"][session_id] = session.get_memory_stats()
        
        return stats