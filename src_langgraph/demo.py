"""
Consolidated Demo for LangGraph RAG Agent Components
Demonstrates LangGraph workflow orchestration with RAG capabilities.
"""

import os
# Disable LangSmith tracing to avoid rate limit errors
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING_ENABLED"] = "false"
from src_langgraph.rag_agent import RAGAgent
from src_langgraph.embedding_manager import EmbeddingManager
from src_langgraph.memory_manager import LangGraphMemoryManager, RAGState
from src_langgraph.document_processor import DocumentProcessor


def demo_rag_agent():
    """Demonstrate LangGraph RAG Agent with workflow orchestration."""
    print("🧠 LangGraph RAG Agent with Workflow Orchestration")
    print("=" * 60)
    print("This example shows how LangGraph enhances RAG with workflow control.")
    print("Workflow: Retrieve → Analyze → Generate/Clarify → Update Memory")
    print()
    
    embedding_manager = EmbeddingManager()
    agent = RAGAgent(embedding_manager, memory_window_size=3)
    
    # Ask a sequence of related questions to demonstrate workflow
    questions = [
        "What is artificial intelligence?",
        "What are the main types of AI?",
        "What is machine learning?",
        "How does machine learning relate to AI?",
        "What is deep learning?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"🤔 Question {i}: {question}")
        print("-" * 50)
        
        # Get answer through LangGraph workflow
        answer = agent.answer_query(question)
        print(f"🤖 Answer: {answer}")
        
        # Show workflow state
        workflow_state = agent.get_workflow_state()
        print(f"📊 Workflow State:")
        print(f"  - Session ID: {workflow_state['session_id']}")
        print(f"  - Memory Turns: {workflow_state['memory_stats'].get('total_turns', 0)}")
        print(f"  - Memory Usage: {workflow_state['memory_stats'].get('memory_usage_percent', 0):.1f}%")
        
        # Show recent conversation history
        history = agent.get_conversation_history()
        if history:
            print("📜 Recent conversation:")
            for turn in history[-2:]:  # Show last 2 turns
                print(f"  Q: {turn['user_message']}")
                print(f"  A: {turn['assistant_response'][:80]}...")
        
        print()
    
    print("=" * 60)
    print("🔄 Testing Workflow with No Relevant Information:")
    print("-" * 50)
    
    # Test a question that might not have relevant chunks
    unclear_question = "What is the meaning of life according to quantum physics?"
    print(f"🤔 Unclear question: {unclear_question}")
    
    answer = agent.answer_query(unclear_question)
    print(f"🤖 Answer: {answer}")
    
    print("\n✅ LangGraph workflow demonstration completed!")
    print("\nKey Advantages:")
    print("• Workflow orchestration with conditional logic")
    print("• Automatic clarification when no relevant info found")
    print("• State-based error handling and recovery")
    print("• Visual workflow traces for debugging")


def demo_embedding_manager():
    """Demonstrate LangGraph Embedding Manager functionality."""
    print("🔍 LangGraph Embedding Manager Demonstration")
    print("=" * 50)
    
    embedding_manager = EmbeddingManager()
    
    # First, create and insert sample document into Pinecone
    print("📄 Creating and inserting sample document into Pinecone...")
    print("-" * 40)
    
    sample_text = """
    This is a sample document about artificial intelligence. 
    AI has become increasingly important in modern technology.
    Machine learning is a subset of AI that focuses on algorithms.
    Deep learning uses neural networks with multiple layers.
    Natural language processing helps computers understand human language.
    Computer vision enables machines to interpret visual information.
    """
    
    # Write sample text to temporary file
    with open("temp_sample.txt", "w") as f:
        f.write(sample_text)
    
    try:
        # Process the document into chunks
        processor = DocumentProcessor(chunk_size=150, chunk_overlap=20)
        chunks = processor.process_file("temp_sample.txt")
        print(f"✅ Created {len(chunks)} chunks from sample document")
        
        # Store chunks in Pinecone
        embedding_manager.store_chunks(chunks)
        print("✅ Stored chunks in Pinecone index")
        
        # Clean up temporary file
        os.remove("temp_sample.txt")
        
    except Exception as e:
        print(f"❌ Error processing/storing document: {str(e)}")
        if os.path.exists("temp_sample.txt"):
            os.remove("temp_sample.txt")
    
    print("\n" + "=" * 50)
    print("🧪 Testing LangGraph Embedding Manager Functions:")
    print("-" * 40)
    
    # Test embedding creation
    test_text = "This is a test sentence for embedding."
    print(f"📝 Test text: {test_text}")
    
    try:
        embedding = embedding_manager.get_embedding(test_text)
        print(f"✅ Created embedding with {len(embedding)} dimensions")
        print(f"📊 First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"❌ Error creating embedding: {str(e)}")
    
    # Test batch embedding
    test_texts = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence."
    ]
    print(f"\n📝 Batch test texts: {test_texts}")
    
    try:
        embeddings = embedding_manager.get_embeddings_batch(test_texts)
        print(f"✅ Created {len(embeddings)} embeddings in batch")
        print(f"📊 First embedding dimensions: {len(embeddings[0])}")
    except Exception as e:
        print(f"❌ Error creating batch embeddings: {str(e)}")
    
    # Test similarity search
    query = "artificial intelligence"
    print(f"\n🔍 Searching for: '{query}'")
    
    try:
        results = embedding_manager.search_similar(query, top_k=3)
        print(f"✅ Found {len(results)} similar chunks:")
        for i, result in enumerate(results, 1):
            score = result.get('score', 0)
            text = result.get('text', '')[:100]
            print(f"  {i}. Score: {score:.3f} | Text: {text}...")
    except Exception as e:
        print(f"❌ Error searching: {str(e)}")
    
    # Get index stats
    try:
        stats = embedding_manager.get_index_stats()
        print(f"\n📊 Index stats: {stats}")
    except Exception as e:
        print(f"❌ Error getting stats: {str(e)}")


def demo_memory_manager():
    """Demonstrate LangGraph Memory Manager functionality."""
    print("🧠 LangGraph Memory Manager Demonstration")
    print("=" * 50)
    
    # Test basic memory
    print("📝 Testing LangGraph Memory (Window Size = 3):")
    print("-" * 40)
    
    memory_manager = LangGraphMemoryManager(default_window_size=3)
    
    # Add conversation turns
    conversations = [
        ("What is Python?", "Python is a programming language."),
        ("What are its features?", "Python has features like readability and extensive libraries."),
        ("How do I install it?", "You can install Python from python.org or use package managers."),
        ("What is machine learning?", "Machine learning is a subset of AI."),
        ("What is deep learning?", "Deep learning uses neural networks with multiple layers.")
    ]
    
    for i, (question, answer) in enumerate(conversations, 1):
        print(f"\n🤔 Adding conversation {i}:")
        print(f"Q: {question}")
        print(f"A: {answer}")
        
        # Show history BEFORE adding
        print(f"📜 History BEFORE adding turn {i}:")
        print(memory_manager.get_formatted_history("session1", 'compact'))
        
        # Add the conversation turn
        memory_manager.add_conversation_turn("session1", question, answer)
        
        # Show history AFTER adding
        print(f"📜 History AFTER adding turn {i}:")
        print(memory_manager.get_formatted_history("session1", 'compact'))
        
        stats = memory_manager.get_session_stats()["sessions"]["session1"]
        print(f"📊 Memory: {stats['total_turns']}/{stats['window_size']} turns")
        
        if i > 1:
            print(f"💡 Note: Turn {i-1} was dropped due to window size limit!")
        print("-" * 40)
    
    # Test multi-session memory manager
    print("\n" + "=" * 60)
    print("🧠 Multi-Session LangGraph Memory Manager Demonstration")
    print("=" * 60)
    
    # Simulate different user sessions
    users = {
        "alice": "user_alice_123",
        "bob": "user_bob_456", 
        "charlie": "user_charlie_789"
    }
    
    print(f"👥 Managing {len(users)} user sessions...")
    
    # Alice's conversation
    print(f"\n👤 {users['alice']} - Python Learning Session:")
    print("-" * 40)
    
    alice_conversation = [
        ("What is Python?", "Python is a high-level programming language known for its simplicity and readability."),
        ("How do I install Python?", "You can download Python from python.org or use package managers like conda or brew."),
        ("What are the best Python libraries?", "Popular Python libraries include NumPy, Pandas, Matplotlib, and TensorFlow.")
    ]
    
    for question, answer in alice_conversation:
        memory_manager.add_conversation_turn(users['alice'], question, answer)
        session = memory_manager.get_session(users['alice'])
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"📊 Memory: {len(session.chat_memory.messages)//2}/{session.k} turns")
    
    # Show overall stats
    print(f"\n📊 Overall Memory Manager Statistics:")
    print("-" * 40)
    overall_stats = memory_manager.get_session_stats()
    print(f"Total Sessions: {overall_stats['total_sessions']}")
    
    for session_id, stats in overall_stats['sessions'].items():
        print(f"\n{session_id}:")
        print(f"  - Turns: {stats['total_turns']}/{stats['window_size']}")
        print(f"  - Usage: {stats['memory_usage_percent']:.1f}%")


def demo_document_processor():
    """Demonstrate LangGraph Document Processor functionality."""
    print("📄 LangGraph Document Processor Demonstration")
    print("=" * 50)
    
    # Create sample text
    sample_text = """
    This is a sample document about artificial intelligence. 
    AI has become increasingly important in modern technology.
    Machine learning is a subset of AI that focuses on algorithms.
    Deep learning uses neural networks with multiple layers.
    Natural language processing helps computers understand human language.
    Computer vision enables machines to interpret visual information.
    """
    
    print("📝 Sample text:")
    print(sample_text.strip())
    print()
    
    # Test different chunk sizes
    chunk_sizes = [100, 150, 200]
    
    for chunk_size in chunk_sizes:
        print(f"🔧 Testing with chunk_size={chunk_size}, chunk_overlap=20:")
        print("-" * 40)
        
        processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=20)
        
        # Write sample text to temporary file
        with open("temp_sample.txt", "w") as f:
            f.write(sample_text)
        
        try:
            # Process the document
            chunks = processor.process_file("temp_sample.txt")
            print(f"✅ Created {len(chunks)} chunks:")
            
            for i, chunk in enumerate(chunks, 1):
                print(f"\nChunk {i}:")
                print(f"Text: {chunk.page_content}")
                print(f"Metadata: {chunk.metadata}")
            
            # Clean up
            os.remove("temp_sample.txt")
            
        except Exception as e:
            print(f"❌ Error processing document: {str(e)}")
            if os.path.exists("temp_sample.txt"):
                os.remove("temp_sample.txt")
        
        print("\n" + "=" * 50)


def demo_workflow_visualization():
    """Demonstrate LangGraph workflow visualization and debugging."""
    print("🔍 LangGraph Workflow Visualization")
    print("=" * 50)
    
    embedding_manager = EmbeddingManager()
    agent = RAGAgent(embedding_manager, memory_window_size=3)
    
    print("📊 Workflow Structure:")
    print("┌─────────────┐    ┌─────────────┐")
    print("│   Retrieve  │───▶│   Analyze   │")
    print("└─────────────┘    └─────────────┘")
    print("                           │")
    print("                           ▼ (conditional routing)")
    print("              ┌─────────────┐    ┌─────────────┐")
    print("              │  Clarify    │    │   Generate  │")
    print("              └─────────────┘    └─────────────┘")
    print("                       │                │")
    print("                       └───────┬───────┘")
    print("                               ▼")
    print("                          ┌─────────────┐")
    print("                          │Update Memory│")
    print("                          └─────────────┘")
    print("                               │")
    print("                               ▼")
    print("                          ┌─────────┐")
    print("                          │   END   │")
    print("                          └─────────┘")
    print()
    
    # Test workflow with different scenarios
    test_cases = [
        ("What is AI?", "Normal query with relevant information"),
        ("What is the meaning of life?", "Query with no relevant information"),
        ("Tell me more about machine learning", "Follow-up query using memory")
    ]
    
    for query, description in test_cases:
        print(f"\n🧪 Testing: {description}")
        print(f"Query: {query}")
        print("-" * 40)
        
        # Execute workflow
        answer = agent.answer_query(query)
        print(f"Answer: {answer}")
        
        # Show workflow state
        state = agent.get_workflow_state()
        print(f"Workflow State:")
        print(f"  - Memory turns: {state['memory_stats'].get('total_turns', 0)}")
        print(f"  - Session ID: {state['session_id']}")


def demo_delete_pinecone_records():
    """Demonstrate deleting all records from Pinecone index."""
    print("🗑️ Delete All Records from Pinecone Index")
    print("=" * 50)
    print("⚠️  WARNING: This will delete ALL records from your Pinecone index!")
    print("This action cannot be undone.")
    print()
    
    # Get confirmation from user
    confirm = input("Are you sure you want to delete ALL records? (yes/no): ").strip().lower()
    
    if confirm != 'yes':
        print("❌ Operation cancelled.")
        return
    
    try:
        embedding_manager = EmbeddingManager()
        
        # Get current stats before deletion
        print("📊 Current index statistics:")
        stats = embedding_manager.get_index_stats()
        print(f"Index info: {stats}")
        print()
        
        # Delete all records (keeps the index structure)
        print("🗑️ Deleting all records...")
        embedding_manager.delete_all_records()
        
        print("✅ Successfully deleted all records from Pinecone index!")
        print("📊 New index statistics:")
        new_stats = embedding_manager.get_index_stats()
        print(f"Index info: {new_stats}")
        
    except Exception as e:
        print(f"❌ Error deleting records: {str(e)}")
        print("Note: You may need to manually delete the index from Pinecone console.")


def demo_create_pinecone_index():
    """Create a new Pinecone index for the RAG agent."""
    import os
    from pinecone import Pinecone, ServerlessSpec
    from dotenv import load_dotenv
    load_dotenv()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "rag-agent-index-openai")
    dimension = int(os.getenv("PINECONE_INDEX_DIM", "1024"))
    if not pinecone_api_key:
        print("❌ PINECONE_API_KEY not found in environment variables")
        return
    print(f"🔧 Creating Pinecone index: {index_name}")
    print(f"📊 Configuration:")
    print(f"   - Dimension: {dimension}")
    print(f"   - Metric: cosine")
    print(f"   - Environment: gcp-starter (free tier)")
    print()
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        existing_indexes = pc.list_indexes()
        if index_name in [index.name for index in existing_indexes]:
            print(f"⚠️  Index '{index_name}' already exists!")
            return
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"✅ Successfully created Pinecone index: {index_name}")
        print(f"📊 Index details:")
        print(f"   - Name: {index_name}")
        print(f"   - Dimension: {dimension}")
        print(f"   - Metric: cosine")
        print(f"   - Type: serverless")
        print("\n⏳ Waiting for index to be ready...")
        import time
        time.sleep(5)
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        print(f"✅ Index is ready! Current stats: {stats}")
    except Exception as e:
        print(f"❌ Error creating index: {str(e)}")
        print("\n💡 Troubleshooting tips:")
        print("1. Check your Pinecone API key")
        print("2. Ensure you have available quota")
        print("3. Try a different index name if this one is taken")


def main():
    """Main menu for selecting demonstrations."""
    print("🚀 LangGraph RAG Agent Components Demo")
    print("=" * 50)
    print("Choose a demonstration:")
    print("1. LangGraph RAG Agent with Workflow")
    print("2. LangGraph Embedding Manager")
    print("3. LangGraph Memory Manager")
    print("4. LangGraph Document Processor")
    print("5. Workflow Visualization")
    print("6. Delete All Pinecone Records")
    print("7. Create Pinecone Index")
    print("8. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-8): ").strip()
            
            if choice == "1":
                print("\n" + "=" * 60)
                demo_rag_agent()
                print("\n" + "=" * 60)
            elif choice == "2":
                print("\n" + "=" * 60)
                demo_embedding_manager()
                print("\n" + "=" * 60)
            elif choice == "3":
                print("\n" + "=" * 60)
                demo_memory_manager()
                print("\n" + "=" * 60)
            elif choice == "4":
                print("\n" + "=" * 60)
                demo_document_processor()
                print("\n" + "=" * 60)
            elif choice == "5":
                print("\n" + "=" * 60)
                demo_workflow_visualization()
                print("\n" + "=" * 60)
            elif choice == "6":
                print("\n" + "=" * 60)
                demo_delete_pinecone_records()
                print("\n" + "=" * 60)
            elif choice == "7":
                print("\n" + "=" * 60)
                demo_create_pinecone_index()
                print("\n" + "=" * 60)
            elif choice == "8":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-8.")
            
            print("\n" + "=" * 50)
            print("Choose another demonstration or exit:")
            print("1. LangGraph RAG Agent with Workflow")
            print("2. LangGraph Embedding Manager")
            print("3. LangGraph Memory Manager")
            print("4. LangGraph Document Processor")
            print("5. Workflow Visualization")
            print("6. Delete All Pinecone Records")
            print("7. Create Pinecone Index")
            print("8. Exit")
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key before running this demo.")
        exit(1)
    
    if not os.getenv("PINECONE_API_KEY"):
        print("❌ PINECONE_API_KEY not found in environment variables")
        print("Please set your Pinecone API key before running this demo.")
        exit(1)
    
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure your Pinecone index is properly set up.") 