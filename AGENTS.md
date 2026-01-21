# AGENTS.md - Guidelines for Agentic Coding Agents

This document provides comprehensive guidelines for agentic coding agents working in the simple_rag repository. It covers build commands, testing, code style conventions, and operational procedures.

## Build, Lint, and Test Commands

### Running the Application
```bash
# Start the Chainlit application
make run

# Stop the application
make stop

# Check if application is running
ps aux | grep chainlit
```

### Vector Database Operations
```bash
# Start Qdrant vector database
make qdrant_run

# Stop Qdrant
make qdrant_stop

# Load data into Qdrant
make qdrant_load

# Check point count in Qdrant
make qdrant_count

# Rebuild Qdrant container
make qdrant_build
```

### LLM Server Operations
```bash
# Start IBM Granite LLM server
make ibm_granite_run

# Stop IBM Granite server
make ibm_granite_stop
```

### Testing
```bash
# Run logging tests
python test_log.py

# Run evaluation tests (requires data files)
python simple_rag_evaluation.py

# Manual testing - send queries to running app
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "test query"}'
```

### Running a Single Test
```bash
# Run specific evaluation test
python -c "
from simple_rag_evaluation import main
main()
"

# Run logging configuration test
python test_log.py

# Evaluate hybrid search performance
python evaluate_hybrid_search.py
```

### Dependencies and Environment
```bash
# Install dependencies
pip install -r requirement.txt

# Check Python version (requires 3.11+)
python --version

# Verify key dependencies
python -c "import chainlit, qdrant_client, sentence_transformers, langchain, rank_bm25"
```

### Hybrid Search Setup
```bash
# Load hybrid data into Qdrant
python hybrid_load_qdrant.py

# Test hybrid search functionality
python evaluate_hybrid_search.py
```

## Code Style Guidelines

### Python Version and Imports
- **Python Version**: 3.11+ (uses modern type hints and features)
- **Import Order**:
  1. Standard library imports
  2. Third-party imports (alphabetical)
  3. Local imports

```python
import json
import logging
import pathlib

import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from mylogger import MyJSONFormatter
```

### Naming Conventions
- **Functions**: `snake_case` (e.g., `context_retreiver`, `setup_logging`)
- **Classes**: `PascalCase` (e.g., `MyJSONFormatter`)
- **Constants**: `UPPER_CASE` (e.g., `VECTOR_TOP_K`, `DEBUG`)
- **Variables**: `snake_case` (e.g., `chat_history`, `qry_vec`)
- **Modules**: `snake_case` (e.g., `simple_rag.py`, `mylogger.py`)

### Type Hints
- Use comprehensive type hints from `typing` module
- Specify return types for all functions
- Use `List`, `Dict`, `Optional` for complex types

```python
from typing import List, Dict, Optional

def context_retreiver(query: str) -> List[Document]:
    # function implementation

def get_handler_by_name(handler_name: str) -> Optional[logging.Handler]:
    # function implementation
```

### Error Handling
- Use specific exception types, not bare `except`
- Log exceptions with context using `logger.exception()`
- Handle expected errors gracefully, let unexpected ones bubble up

```python
try:
    results = client.query_points(...)
    return [Document(hit.payload['text']) for hit in results.points]
except Exception as e:
    logger.error(f"Search error: {e}")
    return []
```

### Logging
- Use structured logging with JSON format
- Include relevant context in log messages
- Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)

```python
logger.debug(f"{inspect_prompt.__name__}", extra={"prompt": f"{x}"})
logger.error(f"Search error: {e}")
```

### Code Structure
- Keep functions focused on single responsibility
- Use descriptive function and variable names
- Add docstrings for complex functions
- Use constants for magic numbers and strings

```python
VECTOR_TOP_K = 5
VECTOR_SCORE_THRESHOLD = 0.7
VECTOR_COLLECTION = 'simple_rag'
```

### Async/Await Patterns
- Use `async`/`await` for Chainlit handlers
- Handle streaming responses properly
- Maintain chat history state

```python
@cl.on_message
async def on_message(message: cl.Message):
    # async implementation
    async for chunk in stream:
        if token := chunk.text:
            await msg.stream_token(token)
```

### Configuration Management
- Use environment variables for sensitive data
- Store configuration in dedicated files
- Use Pydantic for validation

```python
llm = ChatOpenAI(
    base_url=LLM_SERVER,
    api_key=SecretStr(LLM_API_KEY),
    stream_usage=True,
)
```

### Hybrid Search Configuration
- **HYBRID_SEARCH_ENABLED**: Enable/disable hybrid search (default: true)
- **RERANKER_ENABLED**: Enable/disable cross-encoder reranking (default: true)
- **RERANK_TOP_K**: Number of candidates for reranking (default: 20)
- **HYBRID_DENSE_WEIGHT**: Weight for dense vectors in RRF (default: 0.7)
- **HYBRID_SPARSE_WEIGHT**: Weight for sparse vectors in RRF (default: 0.3)

```bash
# Enable hybrid search with custom weights
export HYBRID_SEARCH_ENABLED=true
export HYBRID_DENSE_WEIGHT=0.6
export HYBRID_SPARSE_WEIGHT=0.4

# Enable reranking with more candidates
export RERANKER_ENABLED=true
export RERANK_TOP_K=25
```

### File Organization
- Keep data files in `/data` directory
- Store logs in `/logs` directory
- Backup files go in `/bak` directory
- Configuration files in `/config` directory
- Sparse vocabulary saved as `data/sparse_vocab.json`

### Security Considerations
- Never log sensitive information (API keys, secrets)
- Use `SecretStr` from Pydantic for sensitive configuration
- Validate inputs before processing
- Handle errors without exposing internal details

### Performance Guidelines
- Use async operations for I/O bound tasks
- Implement proper connection pooling for database clients
- Use streaming for large responses
- Cache expensive operations when appropriate

### Testing Approach
- Focus on integration testing for RAG pipeline
- Test vector search accuracy
- Validate LLM response quality
- Check error handling paths

### Git Workflow
- Follow conventional commit messages
- Test changes before committing
- Use feature branches for new functionality
- Keep commits focused and atomic

### Dependencies
- Pin versions in `requirements.txt`
- Keep dependencies minimal and up-to-date
- Test compatibility when upgrading
- Document any special installation requirements

### Documentation
- Update README.md for significant changes
- Document complex algorithms and business logic
- Include usage examples in docstrings
- Maintain this AGENTS.md file

## Operational Notes

- The application requires Qdrant running on localhost:6333
- LLM server (OpenRouter or local) must be accessible
- Vector embeddings use `all-MiniLM-L6-v2` model for dense vectors
- BM25 sparse vectors are used for lexical matching
- Cross-encoder reranking improves result quality
- Chat history is maintained in memory per session
- Debug mode can be enabled with `<debug>` prefix in messages
- Hybrid search automatically falls back to dense-only if sparse data unavailable

## Common Issues and Solutions

1. **Qdrant Connection Failed**: Ensure `make qdrant_run` has been executed
2. **LLM API Errors**: Check API key and endpoint configuration
3. **Import Errors**: Run `pip install -r requirements.txt`
4. **Memory Issues**: Reduce batch sizes for large datasets
5. **Slow Responses**: Check vector search parameters and thresholds
6. **Hybrid Search Not Working**: Ensure sparse vocabulary exists (`data/sparse_vocab.json`)
7. **Reranker Errors**: Check cross-encoder model loading and available memory</content>
<parameter name="filePath">/home/hagusta/workspace/simple_rag/AGENTS.md