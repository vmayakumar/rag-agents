"""
Document Processor for RAG Agent
Handles loading documents and splitting them into chunks for embedding.
"""

import os
import uuid
from typing import List, Dict, Any
from dataclasses import dataclass
import re


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document."""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str


class DocumentProcessor:
    """Processes documents into chunks for embedding and retrieval."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_text_file(self, file_path: str) -> str:
        """
        Load text from a file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Content of the file as string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {str(e)}")
    
    def split_text_into_chunks(self, text: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """
        Split text into chunks using sentence boundaries, with sentence-level overlap.
        
        Args:
            text: The text to split
            metadata: Additional metadata for the chunks
            
        Returns:
            List of DocumentChunk objects
        """
        if metadata is None:
            metadata = {}
        
        # Clean the text
        text = self._clean_text(text)
        
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        
        chunks = []
        chunk_id = 0
        # Calculate overlap: use 1 sentence for every 200 chars of overlap, minimum 1
        overlap_sentences = max(1, self.chunk_overlap // 200)
        
        i = 0
        while i < len(sentences):
            current_chunk = []
            current_len = 0
            start_i = i
            
            # Add sentences until chunk_size is reached
            while i < len(sentences) and current_len + len(sentences[i]) <= self.chunk_size:
                current_chunk.append(sentences[i])
                current_len += len(sentences[i]) + 1  # +1 for space
                i += 1
            
            # If no sentences were added, force add the current sentence
            if not current_chunk and i < len(sentences):
                current_chunk.append(sentences[i])
                i += 1
            
            # Create the chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, chunk_id, metadata))
                chunk_id += 1
            
            # Handle overlap for next chunk
            if i >= len(sentences):
                break  # No more sentences
                
            # Calculate how many sentences to step back
            step_back = min(overlap_sentences, len(current_chunk))
            
            # Ensure we make progress
            if step_back >= len(current_chunk):
                # If overlap would consume entire chunk, reduce it
                step_back = max(0, len(current_chunk) - 1)
            
            # Step back for overlap, but ensure we don't go backwards
            new_i = i - step_back
            if new_i <= start_i:
                # If we can't make progress, break to avoid infinite loop
                break
            i = new_i
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Split on sentence endings followed by space or end of text
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _create_chunk(self, text: str, chunk_id: int, metadata: Dict[str, Any]) -> DocumentChunk:
        """Create a DocumentChunk with proper metadata."""
        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            'chunk_id': chunk_id,
            'chunk_size': len(text)
        })
        
        return DocumentChunk(
            text=text,
            metadata=chunk_metadata,
            chunk_id=str(uuid.uuid4())
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might cause issues, but keep sentence punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)
        return text.strip()
    
    def process_file(self, file_path: str) -> List[DocumentChunk]:
        """
        Process a file into chunks.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of DocumentChunk objects
        """
        text = self.load_text_file(file_path)
        metadata = {
            'source': file_path,
            'file_name': os.path.basename(file_path),
            'file_size': len(text)
        }
        return self.split_text_into_chunks(text, metadata)
