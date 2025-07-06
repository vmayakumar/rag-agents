# RAG Agents: Custom, LangChain, and LangGraph Implementations

This is an educational project demonstrating Retrieval-Augmented Generation (RAG) using three different implementations.

This project provides hands-on examples of how to build intelligent question-answering systems that combine the power of large language models with external knowledge retrieval.

## 🎯 Project Objectives

This project aims to:

- **Compare three RAG implementation approaches**: Custom, LangChain, and LangGraph
- **Provide educational examples** for understanding RAG fundamentals
- **Demonstrate consistent architecture** across different frameworks
- **Show practical implementations** with real-world use cases
- **Enable easy comparison** between different approaches

## 🏗️ Project Structure

The project is organized into three main directories, each containing the same modular structure:

```
rag-agents/
├── requirements.txt
├── src_custom/          # Custom implementation (no frameworks)
│   ├── demo.py
│   ├── document_processor.py
│   ├── embedding_manager.py
│   ├── memory_manager.py
│   └── rag_agent.py
├── src_langchain/       # LangChain implementation
│   ├── demo.py
│   ├── document_processor.py
│   ├── embedding_manager.py
│   ├── memory_manager.py
│   └── rag_agent.py
└── src_langgraph/       # LangGraph implementation
    ├── demo.py
    ├── document_processor.py
    ├── embedding_manager.py
    ├── memory_manager.py
    └── rag_agent.py
```

### 📁 File Descriptions

Each implementation contains the same five core files:

- **`demo.py`**: Comprehensive demonstration with practical examples
- **`document_processor.py`**: Handles document chunking and preprocessing
- **`embedding_manager.py`**: Manages vector operations and Pinecone integration
- **`memory_manager.py`**: Handles conversation history and context management
- **`rag_agent.py`**: Core orchestrator that ties everything together

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Pinecone API key (optional, for vector database operations)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag-agents
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create a .env file
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   echo "PINECONE_API_KEY=your_pinecone_api_key_here" >> .env
   echo "PINECONE_ENVIRONMENT=your_pinecone_environment" >> .env
   ```

### Running the Demos

#### 1. Custom Implementation
```bash
python -m src_custom.demo
```

#### 2. LangChain Implementation
```bash
python -m src_langchain.demo
```

#### 3. LangGraph Implementation
```bash
python -m src_langgraph.demo
```

## 🔧 Implementation Details

### Custom Implementation
- **Purpose**: Learning RAG fundamentals with complete control
- **Best for**: Educational projects, understanding under-the-hood mechanics
- **Features**: Manual orchestration of each RAG step
- **Dependencies**: Minimal (OpenAI, Pinecone only)

### LangChain Implementation
- **Purpose**: Production-ready RAG with rich ecosystem
- **Best for**: Rapid development, enterprise applications
- **Features**: Chain abstractions, pre-built components
- **Dependencies**: LangChain ecosystem

### LangGraph Implementation
- **Purpose**: Advanced workflow orchestration
- **Best for**: Complex workflows, research projects
- **Features**: Stateful workflows, conditional logic
- **Dependencies**: LangGraph, LangChain

## 📊 Key Features

### All Implementations Include:
- ✅ **Vector Database Integration**: Pinecone for efficient similarity search
- ✅ **Memory Management**: Conversation history and context
- ✅ **Document Processing**: Text chunking and preprocessing
- ✅ **Embedding Generation**: OpenAI embeddings for semantic search
- ✅ **RAG Pipeline**: Complete retrieval-augmented generation workflow

### Demo Capabilities:
- 🔍 **Similarity Search**: Find relevant documents using vector embeddings
- 💬 **Conversation Memory**: Maintain context across multiple interactions
- 📝 **Document Processing**: Chunk and process text documents
- 🤖 **LLM Integration**: Generate responses using OpenAI models
- 📊 **Performance Metrics**: Track retrieval and generation performance

## 🛠️ Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
```

### Pinecone Setup (Optional)
1. Create a Pinecone account at [pinecone.io](https://www.pinecone.io/)
2. Create an index with dimension 1536 (for OpenAI embeddings)
3. Set the metric to "cosine"
4. Add your API key and environment to `.env`

## 📚 Usage Examples

### Basic RAG Query
```python
from src_custom.rag_agent import RAGAgent
from src_custom.embedding_manager import EmbeddingManager

# Initialize components
embedding_manager = EmbeddingManager()
agent = RAGAgent(embedding_manager)

# Ask a question
response = agent.answer_query("What is artificial intelligence?")
print(response)
```

### With Memory
```python
# Enable conversation memory
response1 = agent.answer_query("What is machine learning?")
response2 = agent.answer_query("How does it relate to AI?")  # Uses previous context
```

### LangChain Chain Usage
```python
from src_langchain.rag_agent import RAGAgent

# Use LangChain's chain abstraction
response = agent.answer_query_with_chain("What is deep learning?")
```

## 🔍 Understanding the Differences

| Aspect | Custom | LangChain | LangGraph |
|--------|--------|-----------|-----------|
| **Complexity** | Low | Medium | High |
| **Control** | Maximum | Medium | High |
| **Development Speed** | Slow | Fast | Medium |
| **Learning Curve** | Steep | Moderate | Steep |
| **Use Case** | Education | Production | Research |

## 🧪 Testing

### Run All Demos
```bash
# Test all implementations
python -m src_custom.demo
python -m src_langchain.demo
python -m src_langgraph.demo
```

### Individual Component Testing
```bash
# Test embedding manager
python -c "from src_custom.embedding_manager import EmbeddingManager; em = EmbeddingManager(); print('Embedding manager works!')"

# Test memory manager
python -c "from src_custom.memory_manager import ConversationBufferWindowMemory; mem = ConversationBufferWindowMemory(); print('Memory manager works!')"
```

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   ```
   Error: No API key provided
   ```
   **Solution**: Ensure `OPENAI_API_KEY` is set in your `.env` file

2. **Pinecone Connection Error**
   ```
   Error: Pinecone index not found
   ```
   **Solution**: Create a Pinecone index or check your API credentials

3. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'src_custom'
   ```
   **Solution**: Run from the project root directory

4. **Memory Issues**
   ```
   Error: CUDA out of memory
   ```
   **Solution**: Reduce batch sizes or use CPU-only mode

### Debug Mode
```bash
# Enable debug logging
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -m src_custom.demo --debug
```

## 📖 Learning Path

### For Beginners
1. Start with `src_custom/` to understand RAG fundamentals
2. Read through `demo.py` to see practical examples
3. Experiment with different parameters in `rag_agent.py`

### For Intermediate Users
1. Compare `src_custom/` vs `src_langchain/` implementations
2. Understand how LangChain abstracts common patterns
3. Explore the chain creation in `answer_query_with_chain()`

### For Advanced Users
1. Dive into `src_langgraph/` workflow orchestration
2. Study the state management and conditional logic
3. Experiment with custom workflow nodes

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Report Issues**: Use GitHub issues for bugs or feature requests
2. **Submit PRs**: Fork the repo and submit pull requests
3. **Improve Documentation**: Help make the README and code comments clearer
4. **Add Examples**: Share your use cases and implementations



## 📄 License

This project is licensed under the MIT License for anyone to learn from, modify, and reuse with proper attribution. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for providing the embedding and language models
- Pinecone for the vector database infrastructure
- LangChain team for the excellent framework
- LangGraph team for the workflow orchestration capabilities

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: your-email@example.com

---

**Happy RAG-ing! 🚀**

*This project demonstrates the power of combining retrieval and generation for intelligent AI systems.* 