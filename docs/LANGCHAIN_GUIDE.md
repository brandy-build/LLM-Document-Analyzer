# LangChain Integration Guide

## Overview

The LLM Document Analyzer now includes full LangChain integration (v2.1.0), providing enterprise-grade RAG (Retrieval Augmented Generation) capabilities with:

- **Multi-turn conversation** with memory management
- **Persistent vector stores** using Chroma
- **Structured outputs** with Pydantic
- **Multiple LLM providers** (Gemini, OpenAI)
- **Backward compatibility** with original API

## Architecture

### Component Stack

```
┌─────────────────────────────────────────────────────┐
│           Hybrid Analyzer (Main API)               │
│         - Backward compatible interface            │
│         - Supports both traditional & LangChain    │
└─────────────────────────────────────────────────────┘
         ↓                                    ↓
┌──────────────────────┐        ┌──────────────────────────┐
│ Traditional Analyzer │        │ LangChain Analyzer       │
│ (Original API)       │        │ (New RAG Pipeline)       │
└──────────────────────┘        └──────────────────────────┘
                                        ↓
                        ┌───────────────────────────────┐
                        │   LangChain Components        │
                        ├───────────────────────────────┤
                        │ • Document Loaders            │
                        │ • Text Splitters              │
                        │ • Embeddings                  │
                        │ • Vector Store (Chroma)       │
                        │ • Retrieval Chains            │
                        │ • Memory Management           │
                        │ • Output Parsers              │
                        └───────────────────────────────┘
                                        ↓
                        ┌───────────────────────────────┐
                        │   External Services           │
                        ├───────────────────────────────┤
                        │ • Google Gemini 2.5-Flash     │
                        │ • OpenAI GPT-4                │
                        │ • HuggingFace Embeddings      │
                        │ • Chroma Vector Database      │
                        └───────────────────────────────┘
```

## Core Components

### 1. LangChainDocumentAnalyzer

Pure LangChain implementation with RAG pipeline.

```python
from src import LangChainDocumentAnalyzer

# Initialize analyzer
analyzer = LangChainDocumentAnalyzer(
    llm_provider="gemini",              # or "openai"
    embedding_provider="huggingface",   # or "openai"
    vector_store_type="chroma",
    persist_directory="./vector_store"
)

# Load document
documents = analyzer.load_document("contract.pdf")
processed_docs = analyzer.process_documents(documents)

# Create vector store
analyzer.create_vector_store(processed_docs, collection_name="contracts")

# Answer questions
result = analyzer.answer_question("What are the key terms?")
print(result["answer"])
print(result["citations"])

# Conversational Q&A with history
conv_result = analyzer.conversational_qa("Follow-up question?")
print(conv_result["chat_history"])
```

### 2. HybridDocumentAnalyzer

Recommended for most use cases. Provides both traditional and LangChain APIs.

```python
from src import HybridDocumentAnalyzer

# Initialize (uses both analyzers)
analyzer = HybridDocumentAnalyzer(
    use_langchain=True,
    llm_provider="gemini"
)

# Works with either approach
analyzer.load_document("policy.pdf")
answer = analyzer.answer_question("What are the policy exclusions?")

# Access conversational capabilities (LangChain only)
conv_result = analyzer.conversational_qa("Next question?")

# Clear conversation for new topic
analyzer.clear_conversation()

# Cleanup resources
analyzer.cleanup()
```

### 3. Structured Outputs

Pydantic-based structured outputs for reliability.

```python
from src.langchain_integration import (
    DocumentAnalysisOutput,
    DecisionExplanationOutput,
    CitationAnswer
)

# Analysis returns structured Pydantic models
analysis: DocumentAnalysisOutput = {
    "summary": "...",
    "key_points": ["point1", "point2"],
    "sentiment": "positive",
    "entity_types": {"PERSON": ["John"]},
    "recommendations": ["rec1"]
}

# Decisions include reasoning
decision: DecisionExplanationOutput = {
    "decision": "Approve",
    "reasoning": ["reason1", "reason2"],
    "supporting_facts": ["fact1"],
    "confidence_score": 0.92
}
```

## Key Features

### 1. RAG Pipeline

**Document Loading**
```python
# Automatic format detection (PDF, TXT, Markdown)
documents = analyzer.load_document("docs/policy.pdf")
# Returns: List[Document] with metadata
```

**Smart Text Splitting**
```python
# Semantic-aware chunking
processed = analyzer.process_documents(documents)
# Uses RecursiveCharacterTextSplitter
# - Chunk size: 1000 characters
# - Overlap: 100 characters
# - Preserves semantic boundaries
```

**Vector Embeddings**
```python
# Multiple embedding options
analyzer = LangChainDocumentAnalyzer(
    embedding_provider="huggingface"  # all-MiniLM-L6-v2
    # or embedding_provider="openai"   # text-embedding-3-small
)
```

**Persistent Vector Store**
```python
# Create and persist
analyzer.create_vector_store(documents)
analyzer.vector_store.persist()  # Saves to disk

# Retrieve similar documents
retriever = analyzer.retriever
context = retriever.get_relevant_documents("query")
```

### 2. Question Answering

**Single-turn Q&A**
```python
result = analyzer.answer_question("What are the terms?")
# Returns: {
#     "answer": "The terms are...",
#     "citations": [
#         {
#             "content": "chunk text",
#             "page": 1,
#             "source": "contract.pdf",
#             "confidence": 0.85
#         }
#     ]
# }
```

**Multi-turn Conversation**
```python
# First question
result1 = analyzer.conversational_qa("What is the contract term?")

# Follow-up remembers context
result2 = analyzer.conversational_qa("And the renewal clause?")

# Access chat history
history = analyzer.conversation_memory.buffer

# Clear for new conversation
analyzer.clear_conversation()
```

### 3. Document Analysis

```python
# Comprehensive analysis with structured output
analysis = analyzer.analyze_document(documents)
# Returns: {
#     "summary": "...",
#     "key_points": [...],
#     "sentiment": "positive|negative|neutral",
#     "entity_types": {"PERSON": [...], "ORG": [...]},
#     "recommendations": [...]
# }
```

### 4. Decision Explanation

```python
# Explain decisions with reasoning
explanation = analyzer.explain_decision(
    context="Contract analysis context...",
    decision="Recommend approval"
)
# Returns: {
#     "decision": "Recommend approval",
#     "reasoning": ["reason1", "reason2"],
#     "supporting_facts": ["fact1", "fact2"],
#     "confidence_score": 0.89
# }
```

## CLI Interface

### Commands

```bash
# Load document
langchain-cli load contract.pdf --provider gemini

# Ask question
langchain-cli ask "What are the key terms?" --provider gemini

# Multi-turn conversation
langchain-cli converse --provider gemini

# Comprehensive analysis
langchain-cli analyze contract.pdf --provider gemini

# Explain decision
langchain-cli explain "Approve proposal" --context "Based on analysis..."

# Show configuration
langchain-cli config
```

### Examples

```bash
# Legal document Q&A
langchain-cli load contracts/agreement.pdf
langchain-cli ask "What is the termination clause?"
langchain-cli ask "What are the liability limits?"

# Policy analysis
langchain-cli load policies/hr-policy.pdf
langchain-cli converse
> You: What are the vacation policies?
> You: What about sick leave?
> You: exit

# Risk assessment
langchain-cli analyze compliance/regulations.pdf
langchain-cli explain "Flag for legal review" --context "Regulatory analysis"
```

## Configuration

### Environment Variables

```bash
# Required
GOOGLE_API_KEY=your-gemini-api-key
OPENAI_API_KEY=your-openai-api-key

# Optional
LOG_LEVEL=INFO
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
VECTOR_STORE_TYPE=chroma
```

### Secure Configuration

```python
from src import get_config

config = get_config()
gemini_key = config.get_gemini_key()      # From .secrets
openai_key = config.get_openai_key()      # Secure loading
```

## Vector Store Options

### Chroma (Recommended)

**Pros:**
- Embedded, no external service needed
- Fast local operations
- Persistent storage to disk
- Good for development and small-medium datasets

```python
analyzer = LangChainDocumentAnalyzer(
    vector_store_type="chroma",
    persist_directory="./vector_stores"
)
```

**Use Cases:** Single-machine, development, research

### FAISS (In-Memory)

**Future Option** - Can be added for:
- Large-scale document sets
- Performance-critical applications
- Distributed systems

## LLM Providers

### Gemini 2.5-Flash (Recommended)

**Pros:**
- Latest model capabilities
- Fast response times
- Free tier available
- Excellent reasoning

```python
analyzer = LangChainDocumentAnalyzer(
    llm_provider="gemini"
)
```

### OpenAI GPT-4

**Pros:**
- Battle-tested reliability
- Consistent behavior
- Better for structured outputs
- Extensive documentation

```python
analyzer = LangChainDocumentAnalyzer(
    llm_provider="openai"
)
```

## Performance Considerations

### Optimization Tips

1. **Chunk Size**
   - Smaller chunks: More relevant context, higher token cost
   - Larger chunks: Fewer tokens, less specific context
   - Default 1000 chars is balanced

2. **Retriever K**
   - Default k=5 returns top 5 documents
   - Increase for broader context
   - Decrease for focused answers

3. **Embeddings**
   - HuggingFace: Free, local, good quality
   - OpenAI: More expensive, higher quality

4. **Temperature**
   - Lower (0.3): More deterministic, factual
   - Higher (0.9): More creative, varied
   - Default: 0.7 (balanced)

### Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| Load PDF | 0.5-2s | Depends on file size |
| Create embeddings | 2-10s | 100 documents |
| Single Q&A | 1-3s | With LLM call |
| Conversational Q&A | 2-4s | With history |

## Error Handling

```python
from src import LangChainDocumentAnalyzer

try:
    analyzer = LangChainDocumentAnalyzer()
    result = analyzer.answer_question("question")
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Runtime error: {e}")
finally:
    analyzer.cleanup()
```

## Testing

```bash
# Run LangChain integration tests
pytest tests/test_langchain_integration.py -v

# Run with coverage
pytest tests/test_langchain_integration.py --cov=src.langchain_integration

# Specific test
pytest tests/test_langchain_integration.py::TestLangChainDocumentAnalyzer -v
```

## Migration from v2.0 to v2.1

### No Breaking Changes!

Original API still works:

```python
# Old code still works
from src import EnhancedDocumentAnalyzer

analyzer = EnhancedDocumentAnalyzer()
result = analyzer.analyze()
```

### Gradual Migration

```python
# Old approach (still works)
from src import EnhancedDocumentAnalyzer
analyzer = EnhancedDocumentAnalyzer()

# New approach (recommended)
from src import HybridDocumentAnalyzer
analyzer = HybridDocumentAnalyzer(use_langchain=True)

# Pure LangChain (advanced)
from src import LangChainDocumentAnalyzer
analyzer = LangChainDocumentAnalyzer()
```

## Advanced Use Cases

### 1. Multi-Document Analysis

```python
documents = []
for pdf_file in Path("contracts/").glob("*.pdf"):
    docs = analyzer.load_document(str(pdf_file))
    documents.extend(docs)

analyzer.create_vector_store(documents)
result = analyzer.answer_question("Compare terms across all contracts")
```

### 2. Custom Prompts

```python
from langchain.prompts import ChatPromptTemplate

custom_prompt = ChatPromptTemplate.from_template("""
Context: {context}
Question: {question}

You are a legal expert. Provide analysis with specific clause references.
Answer:
""")

# Use custom prompt in chains (advanced)
```

### 3. Streaming Responses

```python
# Future feature - streaming LLM responses
# for real-time user feedback
```

## Troubleshooting

### Vector Store Not Found

```python
# Ensure persist_directory exists
import os
os.makedirs(persist_directory, exist_ok=True)
```

### API Key Issues

```bash
# Check configuration
langchain-cli config

# Ensure .secrets file exists
cat .secrets.example > .secrets
# Edit .secrets with actual API key
```

### Memory Issues with Large Documents

```python
# Reduce chunk size or retriever k
analyzer = LangChainDocumentAnalyzer(
    # ... other params
)
# Then adjust in queries
```

## Additional Resources

- [LangChain Documentation](https://python.langchain.com)
- [Chroma Vector Store](https://www.trychroma.com)
- [Google Gemini API](https://ai.google.dev)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [HuggingFace Sentence Transformers](https://www.sbert.net)

## Support

For issues or questions:
1. Check `SECURITY.md` for credential setup
2. Review test cases in `tests/test_langchain_integration.py`
3. Check application logs (set `LOG_LEVEL=DEBUG`)
4. See GitHub issues or documentation
