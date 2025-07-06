"""
Embedding Manager for RAG Agent using LangChain
Handles converting text to embeddings and storing them using LangChain components.
"""

import os
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class EmbeddingManager:
    """Manages text embeddings and Pinecone vector database operations using LangChain."""
    
    def __init__(self, index_name: str = None, dimension: int = None):
        """
        Initialize the embedding manager with LangChain components.
        
        Args:
            index_name: Name of the Pinecone index to use
            dimension: Dimension of the embedding vectors
        """
        # Initialize OpenAI embeddings using LangChain
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1024
        )
        
        # Initialize Pinecone vector store using LangChain
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "rag-agent-index-openai")
        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings
        )
        
        print(f"Connected to Pinecone index: {self.index_name}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Convert text to embedding using LangChain OpenAI embeddings.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding
        """
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            raise Exception(f"Error creating embedding: {str(e)}")
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple texts to embeddings in batch using LangChain.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            raise Exception(f"Error creating batch embeddings: {str(e)}")
    
    def store_chunks(self, chunks: List[Document]) -> None:
        """
        Store document chunks in Pinecone using LangChain vector store.
        
        Args:
            chunks: List of LangChain Document objects to store
        """
        if not chunks:
            print("No chunks to store")
            return
        
        try:
            # LangChain handles the embedding and storage automatically
            self.vector_store.add_documents(chunks)
            print(f"Stored {len(chunks)} chunks in Pinecone using LangChain")
        except Exception as e:
            raise Exception(f"Error storing documents in Pinecone: {str(e)}")
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using LangChain vector store.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing similar chunks and their scores
        """
        try:
            # LangChain handles the embedding and search automatically
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
            
            # Format results to match our original interface
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'id': doc.metadata.get('chunk_id', 'unknown'),
                    'score': score,
                    'text': doc.page_content,
                    'source': doc.metadata.get('source', 'unknown'),
                    'file_name': doc.metadata.get('file_name', 'unknown'),
                    'chunk_id': doc.metadata.get('chunk_id', 'unknown')
                })
            
            return formatted_results
            
        except Exception as e:
            raise Exception(f"Error searching in Pinecone: {str(e)}")
    
    def delete_all_records(self):
        """Delete all records from the Pinecone index without deleting the index."""
        try:
            import pinecone
            pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index(self.index_name)
            stats = index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            if total_vectors == 0:
                print("Index is already empty.")
                return
            print(f"Found {total_vectors} vectors to delete...")
            index.delete(delete_all=True)
            print(f"Deleted all vectors from index: {self.index_name}")
        except Exception as e:
            raise Exception(f"Error deleting all records: {str(e)}")
    
    def delete_index(self):
        """Delete the Pinecone index (use with caution!)."""
        try:
            # Note: LangChain doesn't provide direct index deletion
            # You would need to use the Pinecone client directly
            print(f"To delete index {self.index_name}, use Pinecone client directly")
        except Exception as e:
            print(f"Error deleting index: {str(e)}")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index."""
        try:
            # Note: LangChain doesn't provide direct stats access
            # You would need to use the Pinecone client directly
            return {"message": "Use Pinecone client directly for index stats"}
        except Exception as e:
            raise Exception(f"Error getting index stats: {str(e)}")