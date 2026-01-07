"""LLM Document Analyzer - Main Package"""

__version__ = "2.0.0"
__author__ = "Document Analysis Team"

from .analyzer import DocumentAnalyzer
from .enhanced_analyzer import EnhancedDocumentAnalyzer
from .secure_config import SecureConfig, get_config
from .models import (
    AnalysisResult,
    DocumentType,
    QuestionAnswer,
    Citation,
    DecisionExplanation,
    DocumentMetadata,
)
from .gemini_client import GeminiClient
from .document_processor import PDFProcessor
from .embeddings import EmbeddingStore

__all__ = [
    "DocumentAnalyzer",
    "EnhancedDocumentAnalyzer",
    "AnalysisResult",
    "DocumentType",
    "QuestionAnswer",
    "Citation",
    "DecisionExplanation",
    "DocumentMetadata",
    "GeminiClient",
    "PDFProcessor",
    "EmbeddingStore",
]
