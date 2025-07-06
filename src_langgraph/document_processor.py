"""
Document Processor for LangGraph RAG Agent
Reuses LangChain document processing components.
"""

from src_langchain.document_processor import DocumentProcessor

# Re-export the LangChain document processor
__all__ = ['DocumentProcessor'] 