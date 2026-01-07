"""Tests for document analyzer"""

import pytest
from unittest.mock import patch, MagicMock
from src.analyzer import DocumentAnalyzer
from src.models import AnalysisRequest, DocumentType, AnalysisResult


@pytest.fixture
def mock_openai():
    """Mock OpenAI client"""
    with patch("src.analyzer.OpenAI") as mock:
        yield mock


@pytest.fixture
def sample_document():
    """Sample document for testing"""
    return """
    The future of artificial intelligence is bright and promising.
    Machine learning models are becoming increasingly sophisticated.
    Natural language processing has revolutionized how we interact with computers.
    Deep learning algorithms are now solving complex problems across industries.
    """


def test_document_analyzer_init():
    """Test DocumentAnalyzer initialization"""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        analyzer = DocumentAnalyzer(api_key="test-key")
        assert analyzer.api_key == "test-key"
        assert analyzer.model == "gpt-4-turbo-preview"


def test_document_analyzer_init_no_key():
    """Test DocumentAnalyzer initialization without API key"""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError):
            DocumentAnalyzer()


def test_detect_document_type_markdown():
    """Test markdown document detection"""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        analyzer = DocumentAnalyzer()
        doc_type = analyzer.detect_document_type("# Title\nSome content")
        assert doc_type == DocumentType.MARKDOWN


def test_detect_document_type_code():
    """Test code document detection"""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        analyzer = DocumentAnalyzer()
        doc_type = analyzer.detect_document_type("def hello():\n    pass")
        assert doc_type == DocumentType.CODE


def test_detect_document_type_email():
    """Test email document detection"""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        analyzer = DocumentAnalyzer()
        doc_type = analyzer.detect_document_type("Subject: Test\nFrom: user@example.com")
        assert doc_type == DocumentType.EMAIL


def test_detect_document_type_text():
    """Test text document detection"""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        analyzer = DocumentAnalyzer()
        doc_type = analyzer.detect_document_type("This is plain text content.")
        assert doc_type == DocumentType.TEXT


@patch("src.analyzer.OpenAI")
def test_analysis_request_creation(mock_openai, sample_document):
    """Test creating analysis request"""
    request = AnalysisRequest(
        content=sample_document,
        document_type=DocumentType.TEXT,
        analyze_sentiment=True,
        include_recommendations=True,
    )
    assert request.content == sample_document
    assert request.document_type == DocumentType.TEXT
    assert request.analyze_sentiment is True


@patch("src.analyzer.DocumentAnalyzer._call_llm")
@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
def test_extract_summary(mock_llm):
    """Test summary extraction"""
    mock_llm.return_value = "This is a test summary."
    analyzer = DocumentAnalyzer()
    summary = analyzer._extract_summary("Test content")
    assert summary == "This is a test summary."
    assert mock_llm.called


@patch("src.analyzer.DocumentAnalyzer._call_llm")
@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
def test_extract_key_points(mock_llm):
    """Test key points extraction"""
    mock_llm.return_value = '[{"text": "Point 1", "importance": 0.9}]'
    analyzer = DocumentAnalyzer()
    points = analyzer._extract_key_points("Test content")
    assert len(points) == 1
    assert points[0].text == "Point 1"
    assert points[0].importance == 0.9


@patch("src.analyzer.DocumentAnalyzer._call_llm")
@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
def test_analyze_sentiment(mock_llm):
    """Test sentiment analysis"""
    mock_llm.return_value = (
        '{"overall_sentiment": "positive", "confidence": 0.95, "emotional_tone": "optimistic"}'
    )
    analyzer = DocumentAnalyzer()
    sentiment = analyzer._analyze_sentiment("Great content!")
    assert sentiment.overall_sentiment == "positive"
    assert sentiment.confidence == 0.95


@patch("src.analyzer.DocumentAnalyzer._call_llm")
@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
def test_extract_entities(mock_llm):
    """Test entity extraction"""
    mock_llm.return_value = '["Apple", "Microsoft", "Google"]'
    analyzer = DocumentAnalyzer()
    entities = analyzer._extract_entities("Companies include Apple and Microsoft")
    assert len(entities) == 3
    assert "Apple" in entities


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
