"""
Embedding Manager for RAG Agent
Handles converting text to embeddings and storing them in Pinecone.
"""

import os
import openai
from typing import List, Dict, Any
from pinecone import Pinecone
from dotenv import load_dotenv
from src_custom.document_processor import DocumentChunk, DocumentProcessor

# Load environment variables
load_dotenv()

class EmbeddingManager:
    """Manages text embeddings and Pinecone vector database operations."""
    
    def __init__(self, index_name: str = None, dimension: int = None):
        """
        Initialize the embedding manager.
        
        Args:
            index_name: Name of the Pinecone index to use
            dimension: Dimension of the embedding vectors (default 3072 for text-embedding-3-large)
        """
        # Initialize OpenAI client
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize Pinecone
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "rag-agent-index-openai")
        self.dimension = dimension or int(os.getenv("PINECONE_INDEX_DIM", "1024"))
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = None
        self._ensure_index_exists()
    
    def _ensure_index_exists(self):
        """Connect to the existing Pinecone index only (do not create)."""
        self.index = self.pc.Index(self.index_name)
        print(f"Connected to Pinecone index: {self.index_name}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Convert text to embedding using OpenAI text-embedding-3-large.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding
        """
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-large",
                input=text,
                dimensions=1024
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Error creating embedding: {str(e)}")
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple texts to embeddings in batch using OpenAI text-embedding-3-large.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-large",
                input=texts,
                dimensions=1024
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise Exception(f"Error creating batch embeddings: {str(e)}")
    
    def store_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Store document chunks in Pinecone with their embeddings.
        
        Args:
            chunks: List of DocumentChunk objects to store
        """
        if not chunks:
            print("No chunks to store")
            return
        
        # Get embeddings for all chunks
        texts = [chunk.text for chunk in chunks]
        embeddings = self.get_embeddings_batch(texts)
        
        # Prepare vectors for Pinecone
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_data = {
                'id': chunk.chunk_id,
                'values': embedding,
                'metadata': {
                    'text': chunk.text,
                    'source': chunk.metadata.get('source', 'unknown'),
                    'file_name': chunk.metadata.get('file_name', 'unknown'),
                    'chunk_id': chunk.metadata.get('chunk_id', i),
                    'chunk_size': chunk.metadata.get('chunk_size', len(chunk.text))
                }
            }
            vectors.append(vector_data)
        
        # Upsert to Pinecone
        try:
            self.index.upsert(vectors=vectors)
            print(f"Stored {len(vectors)} chunks in Pinecone")
        except Exception as e:
            raise Exception(f"Error storing vectors in Pinecone: {str(e)}")
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using semantic search.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing similar chunks and their scores
        """
        # Get embedding for the query
        query_embedding = self.get_embedding(query)
        
        # Search in Pinecone
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Format results
            formatted_results = []
            for match in results['matches']:
                formatted_results.append({
                    'id': match['id'],
                    'score': match['score'],
                    'text': match['metadata']['text'],
                    'source': match['metadata']['source'],
                    'file_name': match['metadata']['file_name'],
                    'chunk_id': match['metadata']['chunk_id']
                })
            
            return formatted_results
            
        except Exception as e:
            raise Exception(f"Error searching in Pinecone: {str(e)}")
    
    def delete_all_records(self):
        """Delete all records from the Pinecone index without deleting the index."""
        try:
            stats = self.index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            if total_vectors == 0:
                print("Index is already empty.")
                return
            print(f"Found {total_vectors} vectors to delete...")
            self.index.delete(delete_all=True)
            print(f"Deleted all vectors from index: {self.index_name}")
        except Exception as e:
            raise Exception(f"Error deleting all records: {str(e)}")
    
    def delete_index(self):
        """Delete the Pinecone index (use with caution!)."""
        try:
            self.pc.delete_index(self.index_name)
            print(f"Deleted Pinecone index: {self.index_name}")
        except Exception as e:
            print(f"Error deleting index: {str(e)}")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index."""
        try:
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            raise Exception(f"Error getting index stats: {str(e)}")
