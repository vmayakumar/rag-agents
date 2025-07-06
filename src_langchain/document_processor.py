"""
Document Processor for RAG Agent using LangChain
Handles loading documents and splitting them into chunks using LangChain components.
"""

import os
import uuid
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.document_loaders import TextLoader


class DocumentProcessor:
    """Processes documents into chunks using LangChain components."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor with LangChain text splitter.
        
        Args:
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            # RAG-optimized separators: prioritize sentence boundaries
            # This strategy is better for RAG because:
            # 1. Complete sentences provide better semantic context for embeddings
            # 2. LLM responses are more coherent with complete thoughts
            # 3. Similar to the original custom implementation's sentence boundary detection
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", ", ", " ", ""]
        )
    
    def load_text_file(self, file_path: str) -> str:
        """
        Load text from a file using LangChain TextLoader.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Content of the file as string
        """
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            return documents[0].page_content
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {str(e)}")
    
    def split_text_into_chunks(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Split text into chunks using LangChain RecursiveCharacterTextSplitter.
        
        Args:
            text: The text to split
            metadata: Additional metadata for the chunks
            
        Returns:
            List of LangChain Document objects
        """
        if metadata is None:
            metadata = {}
        
        # Create a LangChain Document
        document = Document(page_content=text, metadata=metadata)
        
        # Split using LangChain text splitter
        chunks = self.text_splitter.split_documents([document])
        
        # Add chunk-specific metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_id': str(uuid.uuid4()),
                'chunk_size': len(chunk.page_content),
                'source': metadata.get('source', 'unknown')
            })
        
        return chunks
    
    def process_file(self, file_path: str) -> List[Document]:
        """
        Process a file into chunks using LangChain.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of LangChain Document objects
        """
        text = self.load_text_file(file_path)
        metadata = {
            'source': file_path,
            'file_name': os.path.basename(file_path),
            'file_size': len(text)
        }
        return self.split_text_into_chunks(text, metadata)
