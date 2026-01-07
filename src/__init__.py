"""LLM Document Analyzer - Main Package"""

__version__ = "2.1.0"
__author__ = "Document Analysis Team"

from .analyzer import DocumentAnalyzer
from .enhanced_analyzer import EnhancedDocumentAnalyzer
from .hybrid_analyzer import HybridDocumentAnalyzer
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

# Lazy imports for LangChain components (optional dependency)
def _get_langchain_analyzer():
    """Lazy load LangChain analyzer to avoid import errors"""
    try:
        from .langchain_integration import LangChainDocumentAnalyzer
        return LangChainDocumentAnalyzer
    except ImportError:
        return None

def LangChainDocumentAnalyzer(*args, **kwargs):
    """Lazy-loaded LangChainDocumentAnalyzer"""
    LangChain = _get_langchain_analyzer()
    if LangChain is None:
        raise ImportError(
            "LangChain components not available. "
            "Install with: pip install langchain langchain-community langchain-google-genai langchain-openai chromadb"
        )
    return LangChain(*args, **kwargs)

__all__ = [
    "DocumentAnalyzer",
    "EnhancedDocumentAnalyzer",
    "HybridDocumentAnalyzer",
    "LangChainDocumentAnalyzer",
    "AnalysisResult",
    "DocumentType",
    "QuestionAnswer",
    "Citation",
    "DecisionExplanation",
    "DocumentMetadata",
    "GeminiClient",
    "PDFProcessor",
    "EmbeddingStore",
    "SecureConfig",
    "get_config",
]
