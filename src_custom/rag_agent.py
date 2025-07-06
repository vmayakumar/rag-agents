from src_custom.embedding_manager import EmbeddingManager
from src_custom.memory_manager import ConversationBufferWindowMemory, MemoryManager
import openai

class RAGAgent:
    def __init__(self, embedding_manager: EmbeddingManager, llm_model: str = "gpt-3.5-turbo", 
                 memory_window_size: int = 5, session_id: str = "default"):
        self.embedding_manager = embedding_manager
        self.llm_model = llm_model
        self.memory = ConversationBufferWindowMemory(window_size=memory_window_size)
        self.session_id = session_id

    def answer_query(self, query: str, top_k: int = 3, min_score: float = 0.4, 
                    include_memory: bool = True) -> str:
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
            memory_context = self.memory.get_formatted_history("compact")
            if memory_context:
                memory_context = f"\n\nPrevious Conversation:\n{memory_context}\n"
        
        # Step 5: Create prompt with memory context
        prompt = f"Context:\n{context}{memory_context}\n\nQuestion: {query}\nAnswer:"
        
        # Step 6: Generate answer using OpenAI LLM
        response = openai.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        answer = response.choices[0].message.content
        
        # Step 7: Store the conversation turn in memory
        self.memory.add_conversation_turn(query, answer)
        
        return answer
    
    def get_memory_stats(self):
        """Get statistics about the current memory state."""
        return self.memory.get_memory_stats()
    
    def clear_memory(self):
        """Clear the conversation memory."""
        self.memory.clear_memory()
    
    def get_conversation_history(self):
        """Get the current conversation history."""
        return self.memory.get_conversation_history() 