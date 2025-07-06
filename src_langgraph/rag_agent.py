"""
RAG Agent using LangGraph
Implements Retrieval-Augmented Generation using LangGraph workflow orchestration.
"""

from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from src_langgraph.embedding_manager import EmbeddingManager
from src_langgraph.memory_manager import RAGState, LangGraphMemoryManager


class RAGAgent:
    """RAG Agent using LangGraph workflow orchestration."""
    
    def __init__(self, embedding_manager: EmbeddingManager, llm_model: str = "gpt-3.5-turbo",
                 memory_window_size: int = 5, session_id: str = "default"):
        """
        Initialize the LangGraph RAG agent.
        
        Args:
            embedding_manager: The embedding manager for vector search
            llm_model: The OpenAI model to use for generation
            memory_window_size: Number of conversation turns to keep in memory
            session_id: Unique identifier for the conversation session
        """
        self.embedding_manager = embedding_manager
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.memory_manager = LangGraphMemoryManager(default_window_size=memory_window_size)
        self.session_id = session_id
        
        # Create prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "memory_context", "question"],
            template="""Context:
{context}

{memory_context}

Question: {question}
Answer:"""
        )
        
        # Create the LangGraph workflow
        self.workflow = self._create_workflow()
    
    def _create_workflow(self):
        """Create the LangGraph workflow with nodes for RAG pipeline."""
        
        # Define workflow nodes
        def retrieve_node(state: RAGState) -> RAGState:
            """Retrieve relevant chunks from vector database."""
            try:
                query = state["user_query"]
                retrieved_chunks = self.embedding_manager.search_similar(query, top_k=3)
                state["retrieved_chunks"] = retrieved_chunks
                state["error_count"] = 0
            except Exception as e:
                state["error_count"] = state.get("error_count", 0) + 1
                state["retrieved_chunks"] = []
            return state
        
        def analyze_node(state: RAGState) -> RAGState:
            """Analyze retrieval results and determine confidence."""
            chunks = state["retrieved_chunks"]
            
            if not chunks:
                state["confidence"] = 0.0
                state["needs_clarification"] = True
            elif len(chunks) < 2:
                state["confidence"] = 0.3
                state["needs_clarification"] = False
            else:
                state["confidence"] = 0.8
                state["needs_clarification"] = False
            
            return state
        
        def generate_node(state: RAGState) -> RAGState:
            """Generate answer using LLM."""
            try:
                chunks = state["retrieved_chunks"]
                query = state["user_query"]
                
                # Build context from chunks
                if chunks:
                    context = "\n".join([chunk['text'] for chunk in chunks])
                else:
                    context = "No relevant information found."
                
                # Get conversation memory
                memory_context = self.memory_manager.get_formatted_history(
                    self.session_id, "compact"
                )
                if memory_context:
                    memory_context = f"Previous Conversation:\n{memory_context}\n"
                else:
                    memory_context = ""
                
                # Generate answer
                prompt = self.prompt_template.format(
                    context=context,
                    memory_context=memory_context,
                    question=query
                )
                
                response = self.llm.invoke(prompt)
                state["generated_answer"] = response.content
                state["error_count"] = 0
                
            except Exception as e:
                state["generated_answer"] = f"Error generating answer: {str(e)}"
                state["error_count"] = state.get("error_count", 0) + 1
            
            return state
        
        def update_memory_node(state: RAGState) -> RAGState:
            """Update conversation memory."""
            try:
                query = state["user_query"]
                answer = state["generated_answer"]
                
                # Add to memory
                self.memory_manager.add_conversation_turn(
                    self.session_id, query, answer
                )
                
                # Update state with current conversation history
                state["conversation_history"] = self.memory_manager.get_conversation_history(
                    self.session_id
                )
                
            except Exception as e:
                state["error_count"] = state.get("error_count", 0) + 1
            
            return state
        
        def clarify_node(state: RAGState) -> RAGState:
            """Generate clarification request when no relevant information is found."""
            state["generated_answer"] = (
                "I couldn't find relevant information for your question. "
                "Could you please rephrase or provide more specific details?"
            )
            return state
        
        # Create the workflow
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("analyze", analyze_node)
        workflow.add_node("generate", generate_node)
        workflow.add_node("update_memory", update_memory_node)
        workflow.add_node("clarify", clarify_node)
        
        # Add entrypoint edge
        workflow.add_edge("__start__", "retrieve")
        
        # Add edges
        workflow.add_edge("retrieve", "analyze")
        workflow.add_edge("generate", "update_memory")
        workflow.add_edge("update_memory", END)
        
        # Add conditional edge for clarification
        def route_after_analyze(state: RAGState) -> str:
            if state["needs_clarification"]:
                return "clarify"
            return "generate"
        
        workflow.add_conditional_edges("analyze", route_after_analyze)
        workflow.add_edge("clarify", "update_memory")
        
        return workflow.compile()
    
    def answer_query(self, query: str, top_k: int = 3, min_score: float = 0.5) -> str:
        """
        Answer a query using LangGraph RAG workflow.
        
        Args:
            query: The user's question
            top_k: Number of top chunks to retrieve
            min_score: Minimum similarity score threshold
            
        Returns:
            Generated answer
        """
        try:
            # Initialize state
            initial_state = RAGState(
                user_query=query,
                retrieved_chunks=[],
                generated_answer="",
                conversation_history=self.memory_manager.get_conversation_history(self.session_id),
                confidence=0.0,
                needs_clarification=False,
                error_count=0,
                session_id=self.session_id
            )
            
            # Execute workflow
            final_state = self.workflow.invoke(initial_state)
            
            return final_state["generated_answer"]
            
        except Exception as e:
            return f"Error in workflow execution: {str(e)}"
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the current memory state."""
        return self.memory_manager.get_session_stats()["sessions"].get(self.session_id, {})
    
    def clear_memory(self) -> None:
        """Clear the conversation memory."""
        self.memory_manager.clear_session(self.session_id)
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the current conversation history."""
        return self.memory_manager.get_conversation_history(self.session_id)
    
    def get_workflow_state(self) -> Dict[str, Any]:
        """Get the current workflow state for debugging."""
        return {
            "session_id": self.session_id,
            "memory_stats": self.get_memory_stats(),
            "conversation_history": self.get_conversation_history()
        } 