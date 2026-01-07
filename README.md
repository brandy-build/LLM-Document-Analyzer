# LLM Document Analyzer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/brandy-build/LLM-Document-Analyzer)](https://github.com/brandy-build/LLM-Document-Analyzer)

A production-ready Python framework for intelligent document analysis using AI language models. Combines LangChain RAG with multi-provider LLM support for enterprise-grade document processing.

**Latest Version**: 2.1.0 | **Status**: Production-Ready | **Security**: Enterprise-Grade

## ✨ Features

### Core Capabilities
- **Multi-Format Document Support**: Process PDF, TXT, and Markdown files
- **Semantic Embeddings**: Advanced semantic search using all-MiniLM-L6-v2 model
- **AI-Powered Q&A**: Ask questions and get answers with source citations
- **Decision Explanation**: Understand the reasoning behind analytical decisions
- **Document Summarization**: Generate concise, accurate summaries
- **Sentiment Analysis**: Analyze document tone and sentiment
- **Key Point Extraction**: Identify and extract main ideas
- **Named Entity Recognition**: Extract people, organizations, locations, and more

### AI Provider Support
- **Google Gemini** (Default): Primary AI provider - fast and cost-effective
- **OpenAI GPT-4**: Optional fallback provider
- **Dual Provider Support**: Automatically handles provider failures

### Security Features
- **Secure Credential Management**: API keys stored in git-ignored `.secrets` file
- **Automatic Validation**: Credentials validated on startup
- **Enterprise-Grade Protection**: OWASP best practices implemented
- **Team-Friendly**: Templates for new team members

## 📋 Prerequisites

- **Python 3.10+**
- **pip** (Python package manager)
- **Google Gemini API key** (recommended)
- **OpenAI API key** (optional)

## 🚀 Quick Start

### 1. Setup Project
```bash
cd "YT wave"
pip install -r requirements.txt
```

### 2. Configure Credentials
```bash
# Copy secrets template
cp .secrets.example .secrets

# Edit .secrets with your actual API keys
nano .secrets  # or your editor
```

Add your API keys:
```
GOOGLE_API_KEY=your_actual_gemini_key_here
OPENAI_API_KEY=your_actual_openai_key_here (optional)
```

### 3. Validate Setup
```bash
python src/secure_config.py
```

Expected output:
```
[OK] All required credentials are configured
[OK] Configured providers:
  - gemini
```

### 4. Start Using!

**Analyze a document:**
```bash
python -m src.enhanced_cli analyze --document policy.pdf
```

**Ask questions with citations:**
```bash
python -m src.enhanced_cli ask --document policy.pdf
```

**Explain decisions:**
```bash
python -m src.enhanced_cli explain --document policy.pdf
```

## 📖 Usage

### Command Line Interface

#### Analyze Document
```bash
python -m src.enhanced_cli analyze --document path/to/document.pdf
```
Outputs: Summary, sentiment analysis, key points, entities

#### Interactive Q&A
```bash
python -m src.enhanced_cli ask --document policy.pdf
```
Starts an interactive session with source citations

#### Explain Decision
```bash
python -m src.enhanced_cli explain --document policy.pdf --question "Why was this approved?"
```

#### Extract PDF Text
```bash
python -m src.enhanced_cli extract-pdf --document file.pdf --output extracted.txt
```

### Python API

#### Document Analysis
```python
from src.enhanced_analyzer import EnhancedDocumentAnalyzer

analyzer = EnhancedDocumentAnalyzer()
analyzer.load_document("policy.pdf")
result = analyzer.analyze()

print(f"Summary: {result.summary}")
print(f"Sentiment: {result.sentiment}")
print(f"Key Points: {[kp.text for kp in result.key_points]}")
```

#### Q&A with Citations
```python
analyzer = EnhancedDocumentAnalyzer()
analyzer.load_document("policy.pdf")
analyzer.build_embedding_index()

answer = analyzer.answer_question("What are eligibility requirements?")
print(f"Answer: {answer['answer']}")
for citation in answer['citations']:
    print(f"  Source (Page {citation['page']}): {citation['content']}")
```

#### Get Secure Config
```python
from src.secure_config import get_config

config = get_config()
gemini_key = config.get_gemini_key()  # Securely loaded
gemini_model = config.get_gemini_model()
```

## 🏗️ Project Structure

```
.
├── src/                           # Source code
│   ├── analyzer.py               # Original OpenAI analyzer
│   ├── enhanced_analyzer.py      # Enhanced with Gemini & embeddings
│   ├── models.py                 # Pydantic data models
│   ├── cli.py                    # Original CLI
│   ├── enhanced_cli.py           # Enhanced CLI
│   ├── gemini_client.py          # Gemini API wrapper
│   ├── document_processor.py     # PDF/text processing
│   ├── embeddings.py             # Vector search engine
│   ├── secure_config.py          # Secure credential loader
│   └── __init__.py               # Package exports
├── tests/                         # Test suite
│   ├── test_analyzer.py          # 11 unit tests
│   ├── conftest.py               # Pytest config
│   └── __init__.py
├── docs/
│   └── API.md                    # API reference
├── .secrets                       # ACTUAL CREDENTIALS (git-ignored)
├── .secrets.example               # Template for team
├── .env                           # Config template
├── .env.example                   # Config reference
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Dependencies
├── pyproject.toml                # Project config
├── README.md                      # This file
├── SECURITY.md                    # Security guide
└── SECURITY_CHECKLIST.md          # Implementation verification
```

## 🔐 Security Setup

### For New Team Members

```bash
# 1. Copy secrets template
cp .secrets.example .secrets

# 2. Edit with your API keys
nano .secrets

# 3. Verify it's git-ignored
git status  # Should NOT show .secrets

# 4. Start using
python src/secure_config.py  # Validate
```

### Credential Management

- **`.secrets`** - Actual API keys (git-ignored) ⚠️ Never commit
- **`.secrets.example`** - Template for new team members
- **`.env`** - Non-sensitive configuration only

See [SECURITY.md](SECURITY.md) for comprehensive security guide.

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v --cov=src

# Run specific test file
pytest tests/test_analyzer.py -v

# Check coverage
pytest tests/ --cov=src --cov-report=html
```

## 🎨 Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type checking
mypy src/
```

## ⚙️ Configuration

### Environment Variables

Edit `.env` or `.secrets` for configuration:

```bash
# API Keys (in .secrets - NEVER in .env)
GOOGLE_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key

# Models
GEMINI_MODEL=gemini-2.5-flash
OPENAI_MODEL=gpt-4-turbo-preview

# Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
TEMPERATURE=0.7

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2
USE_EMBEDDINGS=true

# Other
LOG_LEVEL=INFO
DEFAULT_PROVIDER=gemini
```

## 📚 API Reference

See [docs/API.md](docs/API.md) for detailed API documentation.

### Key Classes

**EnhancedDocumentAnalyzer**
- `load_document(file_path)` - Load documents
- `build_embedding_index()` - Create embeddings
- `analyze()` - Full analysis
- `answer_question(question)` - Q&A with citations
- `explain_decision(...)` - Decision reasoning

**GeminiClient**
- `generate_text(prompt, ...)` - Generate text
- `answer_question(question, context)` - Answer with context
- `summarize_with_key_points(text)` - Extract summary

**PDFProcessor**
- `load_document(file_path)` - Load any format
- `chunk_text(text, ...)` - Split into chunks

**EmbeddingStore**
- `add_embedding(chunk, embedding, ...)` - Store vectors
- `semantic_search(query, top_k)` - Find similar chunks

**SecureConfig**
- `get_gemini_key()` - Get API key securely
- `get_openai_key()` - Get optional key
- `validate_credentials()` - Check setup

## 📊 Version History

**v2.0.0** (Current)
- ✅ PDF support with embeddings
- ✅ Google Gemini AI integration
- ✅ Semantic Q&A with citations
- ✅ Decision explanation features
- ✅ Secure credential management
- ✅ Dual provider support

**v1.0.0**
- OpenAI GPT-4 integration
- Basic document analysis
- Sentiment & entity extraction

## 🤝 Contributing

1. Create feature branch
2. Add tests for new features
3. Run: `pytest tests/ -v`
4. Format: `black src/ tests/`
5. Lint: `ruff check src/ tests/`
6. Submit PR

## 📄 License

MIT License

## 🆘 Troubleshooting

### Credentials Not Found
```bash
# Verify .secrets exists
test -f .secrets && echo "OK" || echo "Missing"

# Validate setup
python src/secure_config.py
```

### PDF Processing Issues
- Ensure PDF not corrupted
- Try with `.txt` file first
- Check available disk space

### Slow Embeddings
- First-time download takes minutes
- Requires 2GB+ RAM
- Check internet connection

### API Rate Limits
- Wait before retrying (free tier)
- Check usage: https://ai.google.dev/usage
- Consider paid plan for production

## 📚 Documentation

| File | Purpose |
|------|---------|
| [README.md](README.md) | This file - project overview |
| [SECURITY.md](SECURITY.md) | Security guide & best practices |
| [SECURITY_CHECKLIST.md](SECURITY_CHECKLIST.md) | Implementation verification |
| [docs/API.md](docs/API.md) | Detailed API reference |
| [.secrets.example](.secrets.example) | Credentials template |

## Quick Commands

```bash
# Setup
pip install -r requirements.txt
cp .secrets.example .secrets

# Use
python -m src.enhanced_cli analyze --document file.pdf
python -m src.enhanced_cli ask --document file.pdf

# Test
pytest tests/ -v

# Validate
python src/secure_config.py
```

## Support

- Documentation: See [SECURITY.md](SECURITY.md) and [docs/API.md](docs/API.md)
- Issues: Create an issue in repository
- Security: See [SECURITY.md](SECURITY.md) for reporting

---

**Status**: ✅ Production-Ready  
**Security**: ✅ Enterprise-Grade  
**Last Updated**: January 7, 2026  
**Ready for Deployment**: Yes
