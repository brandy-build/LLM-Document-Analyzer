# Copilot Instructions for LLM Document Analyzer

## Project Overview
This is an LLM-Based Document Analyzer - a Python application that uses language models to analyze documents with features like summarization, sentiment analysis, key point extraction, and recommendations.

## Key Files
- **src/analyzer.py**: Main DocumentAnalyzer class
- **src/models.py**: Pydantic data models
- **src/cli.py**: Command-line interface
- **tests/test_analyzer.py**: Unit tests
- **README.md**: Comprehensive documentation
- **docs/API.md**: API reference

## Development Guidelines

### Code Style
- Use Black for formatting
- Follow PEP 8 conventions
- Use type hints throughout
- Max line length: 100 characters

### Testing
- Write unit tests for new features
- Use pytest for testing
- Aim for > 80% coverage
- Mock external API calls

### Dependencies
- Keep dependencies minimal
- Use requirements.txt for core dependencies
- Use pyproject.toml for optional dev/docs dependencies

### LLM Integration
- All LLM calls go through DocumentAnalyzer._call_llm()
- Always handle JSON parsing errors gracefully
- Log analysis progress and timing
- Implement retry logic for failed requests

## Setup for Development

1. **Environment Setup**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"
   ```

2. **Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

3. **Running Tests**
   ```bash
   pytest tests/ -v --cov=src
   ```

4. **Code Quality**
   ```bash
   black src/ tests/
   ruff check src/ tests/
   mypy src/
   ```

## Common Tasks

### Adding a New Analysis Feature
1. Add method to DocumentAnalyzer
2. Update models.py with result structures
3. Add tests in tests/test_analyzer.py
4. Update CLI if needed in src/cli.py
5. Document in docs/API.md

### Handling Errors
- Log all errors with proper level (INFO, WARNING, ERROR)
- Raise ValueError for configuration issues
- Raise custom exceptions for analysis failures
- Always provide user-friendly error messages

### Performance Tips
- Cache analyzer instances when processing multiple documents
- Consider chunking very large documents
- Use lower temperature for faster responses
- Monitor OpenAI API usage

## Extension Points
- Add new document type detectors
- Implement custom analysis methods
- Add support for additional LLM providers
- Create API endpoints for remote analysis
- Add database integration for result storage
