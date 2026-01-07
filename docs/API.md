# API Documentation

## DocumentAnalyzer

The main class for analyzing documents using LLM capabilities.

### Initialization

```python
from src.analyzer import DocumentAnalyzer

analyzer = DocumentAnalyzer(
    api_key="sk-your-api-key",
    model="gpt-4-turbo-preview",
    temperature=0.7
)
```

**Parameters:**
- `api_key` (str, optional): OpenAI API key. Defaults to `OPENAI_API_KEY` environment variable
- `model` (str): Model name. Defaults to `gpt-4-turbo-preview`
- `temperature` (float): Temperature for responses (0-1). Defaults to 0.7

### Methods

#### analyze(request: AnalysisRequest) -> AnalysisResult

Performs comprehensive analysis on a document.

**Parameters:**
- `request` (AnalysisRequest): Analysis request configuration

**Returns:**
- `AnalysisResult`: Complete analysis results

**Example:**
```python
from src.models import AnalysisRequest

request = AnalysisRequest(
    content="Your document here",
    analyze_sentiment=True,
    include_recommendations=True
)

result = analyzer.analyze(request)
print(result.summary)
```

#### detect_document_type(content: str) -> DocumentType

Detects the type of document based on its content.

**Parameters:**
- `content` (str): Document content

**Returns:**
- `DocumentType`: Detected type (PDF, TEXT, MARKDOWN, CODE, EMAIL, UNKNOWN)

**Example:**
```python
doc_type = analyzer.detect_document_type("# Markdown Title\nContent")
print(doc_type)  # DocumentType.MARKDOWN
```

## Data Models

### AnalysisRequest

Configuration for document analysis.

**Fields:**
```python
class AnalysisRequest(BaseModel):
    content: str                          # Document content
    document_type: Optional[DocumentType] # Type hint
    analyze_sentiment: bool               # Default: True
    max_summary_length: int               # Default: 500
    include_recommendations: bool         # Default: True
```

### AnalysisResult

Complete analysis output.

**Fields:**
```python
class AnalysisResult(BaseModel):
    document_type: DocumentType       # Detected type
    document_length: int              # Characters
    summary: str                      # Summarized content
    key_points: List[KeyPoint]        # Important points
    sentiment: Optional[SentimentAnalysis]
    entities: List[str]               # Named entities
    language: str                     # Default: 'en'
    processing_time: float            # Seconds
    confidence_score: float           # 0-1
    recommendations: List[str]        # Suggestions
```

### KeyPoint

Individual key point from analysis.

**Fields:**
```python
class KeyPoint(BaseModel):
    text: str              # Point text
    importance: float      # Score 0-1
    category: Optional[str] # Point category
```

### SentimentAnalysis

Sentiment analysis results.

**Fields:**
```python
class SentimentAnalysis(BaseModel):
    overall_sentiment: str     # 'positive', 'negative', 'neutral'
    confidence: float          # 0-1
    emotional_tone: str        # Description
```

### DocumentType

Enum for document types.

**Values:**
- `PDF`
- `TEXT`
- `MARKDOWN`
- `CODE`
- `EMAIL`
- `UNKNOWN`

## Error Handling

### Common Exceptions

**ValueError**
- Raised when API key is missing
- Raised when invalid configuration provided

**OpenAIError**
- Raised when API call fails
- Check your API key and rate limits

**JSONDecodeError**
- Raised when LLM response isn't valid JSON
- Analyzer logs warning and returns partial results

### Example Error Handling

```python
try:
    result = analyzer.analyze(request)
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Analysis failed: {e}")
```

## CLI Reference

### Commands

#### analyze
Analyze a document file.

```bash
python -m src.cli analyze <document_path> [options]
```

**Options:**
- `--doc-type`: Document type hint (auto, pdf, text, markdown, code, email)
- `--output`: Output file path (JSON)
- `--sentiment/--no-sentiment`: Include sentiment analysis
- `--recommendations/--no-recommendations`: Include recommendations

**Example:**
```bash
python -m src.cli analyze myfile.txt --output results.json --sentiment
```

#### quick-analyze
Quick analysis from text input.

```bash
python -m src.cli quick-analyze --text "Your text"
```

#### version
Show version information.

```bash
python -m src.cli version
```

## Best Practices

1. **Reuse Analyzer Instance**
   ```python
   analyzer = DocumentAnalyzer()
   # Use for multiple requests
   for doc in documents:
       result = analyzer.analyze(doc)
   ```

2. **Handle Large Documents**
   ```python
   # Consider chunking very large documents
   if len(content) > 50000:
       # Process in chunks
       pass
   ```

3. **Monitor Costs**
   - Use lower temperature for faster responses
   - Batch similar requests
   - Monitor API usage

4. **Error Recovery**
   ```python
   from tenacity import retry, stop_after_attempt
   
   @retry(stop=stop_after_attempt(3))
   def analyze_with_retry(analyzer, request):
       return analyzer.analyze(request)
   ```

## Rate Limits

Default OpenAI rate limits apply:
- Check [OpenAI documentation](https://platform.openai.com/docs/guides/rate-limits) for current limits
- Implement exponential backoff for retries
- Monitor usage in OpenAI dashboard

## Examples

### Full Analysis Example

```python
from src.analyzer import DocumentAnalyzer
from src.models import AnalysisRequest

# Initialize
analyzer = DocumentAnalyzer()

# Prepare document
with open("document.txt", "r") as f:
    content = f.read()

# Analyze
request = AnalysisRequest(
    content=content,
    analyze_sentiment=True,
    max_summary_length=300,
    include_recommendations=True
)

result = analyzer.analyze(request)

# Use results
print(f"Type: {result.document_type.value}")
print(f"Summary: {result.summary}")
print(f"Sentiment: {result.sentiment.overall_sentiment}")
print(f"Confidence: {result.confidence_score}")
print(f"Time: {result.processing_time:.2f}s")

for point in result.key_points:
    print(f"- {point.text} (importance: {point.importance})")

for rec in result.recommendations:
    print(f"→ {rec}")
```

### Batch Processing Example

```python
from pathlib import Path
import json

analyzer = DocumentAnalyzer()
results = {}

for doc_file in Path("documents").glob("*.txt"):
    with open(doc_file) as f:
        content = f.read()
    
    request = AnalysisRequest(content=content)
    result = analyzer.analyze(request)
    results[doc_file.name] = result.model_dump()

# Save results
with open("batch_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

**Last Updated:** 2024-01-07
