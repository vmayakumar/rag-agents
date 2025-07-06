"""
Embedding Manager for LangGraph RAG Agent
Reuses LangChain embedding management components.
"""

from src_langchain.embedding_manager import EmbeddingManager

# Re-export the LangChain embedding manager
__all__ = ['EmbeddingManager'] 