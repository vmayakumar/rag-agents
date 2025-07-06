"""
RAG Agent using LangChain
Implements Retrieval-Augmented Generation using LangChain components.
"""

from src_langchain.embedding_manager import EmbeddingManager
from src_langchain.memory_manager import MemoryManager
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from typing import List, Dict, Any


class RAGAgent:
    """RAG Agent using LangChain components for retrieval and generation."""
    
    def __init__(self, embedding_manager: EmbeddingManager, llm_model: str = "gpt-3.5-turbo",
                 memory_window_size: int = 5, session_id: str = "default"):
        """
        Initialize the RAG agent with LangChain components.
        
        Args:
            embedding_manager: The embedding manager for vector search
            llm_model: The OpenAI model to use for generation
            memory_window_size: Number of conversation turns to keep in memory
            session_id: Unique identifier for the conversation session
        """
        self.embedding_manager = embedding_manager
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.memory_manager = MemoryManager(default_window_size=memory_window_size)
        self.session_id = session_id
        
        # Create a custom prompt template that includes memory
        self.prompt_template = PromptTemplate(
            input_variables=["context", "memory_context", "question"],
            template="""Context:
{context}

{memory_context}

Question: {question}
Answer:"""
        )
    
    def answer_query(self, query: str, top_k: int = 3, min_score: float = 0.5, 
                    include_memory: bool = True) -> str:
        """
        Answer a query using LangChain RAG pipeline with memory.
        
        Args:
            query: The user's question
            top_k: Number of top chunks to retrieve
            min_score: Minimum similarity score threshold
            include_memory: Whether to include conversation memory
            
        Returns:
            Generated answer or fallback message
        """
        try:
            # Step 1: Retrieve relevant chunks
            retrieved_chunks = self.embedding_manager.search_similar(query, top_k=top_k)
            
            if not retrieved_chunks:
                return "Sorry, I could not find relevant information."
            
            # Step 2: Filter by similarity score
            relevant_chunks = [chunk for chunk in retrieved_chunks if chunk.get('score', 0) >= min_score]
            
            if not relevant_chunks:
                return "Sorry, I could not find relevant information."
            
            # Step 3: Build context from relevant chunks
            context = "\n".join([chunk['text'] for chunk in relevant_chunks])
            
            # Step 4: Include conversation memory if requested
            memory_context = ""
            if include_memory:
                memory_context = self.memory_manager.get_formatted_history(self.session_id, "compact")
                if memory_context:
                    memory_context = f"Previous Conversation:\n{memory_context}\n"
            
            # Step 5: Generate answer using LangChain
            prompt = self.prompt_template.format(
                context=context, 
                memory_context=memory_context, 
                question=query
            )
            response = self.llm.invoke(prompt)
            
            answer = response.content
            
            # Step 6: Store the conversation turn in memory
            self.memory_manager.add_conversation_turn(self.session_id, query, answer)
            
            return answer
            
        except Exception as e:
            return f"Error occurred: {str(e)}"
    
    def answer_query_with_chain(self, query: str, top_k: int = 3, 
                               include_memory: bool = True) -> str:
        """
        Alternative method using LangChain RetrievalQA chain with memory.
        This demonstrates a more integrated LangChain approach.
        
        Args:
            query: The user's question
            top_k: Number of top chunks to retrieve
            include_memory: Whether to include conversation memory
            
        Returns:
            Generated answer
        """
        try:
            # Create a simple retriever function
            def retrieve_docs(query: str, k: int = top_k):
                results = self.embedding_manager.search_similar(query, top_k=k)
                # Convert to LangChain Document format
                docs = []
                for result in results:
                    doc = Document(
                        page_content=result['text'],
                        metadata={
                            'source': result['source'],
                            'score': result['score'],
                            'chunk_id': result['chunk_id']
                        }
                    )
                    docs.append(doc)
                return docs
            
            # Create RetrievalQA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retrieve_docs,
                return_source_documents=True
            )
            
            # Get answer
            result = qa_chain({"query": query})
            answer = result["result"]
            
            # Store the conversation turn in memory
            self.memory_manager.add_conversation_turn(self.session_id, query, answer)
            
            return answer
            
        except Exception as e:
            return f"Error occurred: {str(e)}"
    
    def get_memory_stats(self):
        """Get statistics about the current memory state."""
        return self.memory_manager.get_session_stats()["sessions"].get(self.session_id, {})
    
    def clear_memory(self):
        """Clear the conversation memory."""
        self.memory_manager.clear_session(self.session_id)
    
    def get_conversation_history(self):
        """Get the current conversation history."""
        return self.memory_manager.get_conversation_history(self.session_id)