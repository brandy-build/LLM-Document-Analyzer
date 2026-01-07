"""Data models for document analysis"""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Supported document types"""

    PDF = "pdf"
    TEXT = "text"
    MARKDOWN = "markdown"
    CODE = "code"
    EMAIL = "email"
    UNKNOWN = "unknown"


class SentimentAnalysis(BaseModel):
    """Sentiment analysis results"""

    overall_sentiment: str = Field(
        ..., description="Overall sentiment: positive, negative, neutral"
    )
    confidence: float = Field(..., description="Confidence score 0-1")
    emotional_tone: str = Field(..., description="Emotional tone of the document")


class KeyPoint(BaseModel):
    """Key point extracted from document"""

    text: str = Field(..., description="The key point text")
    importance: float = Field(..., description="Importance score 0-1")
    category: Optional[str] = Field(None, description="Category of the key point")


class Citation(BaseModel):
    """Citation reference"""

    text: str = Field(..., description="Source text")
    chunk_id: int = Field(..., description="Chunk identifier")
    page: Optional[int] = Field(None, description="Page number if applicable")
    confidence: float = Field(default=1.0, description="Confidence score 0-1")


class QuestionAnswer(BaseModel):
    """Question and answer with citations"""

    question: str = Field(..., description="The question asked")
    answer: str = Field(..., description="The answer provided")
    citations: List[Citation] = Field(default_factory=list, description="Source citations")
    confidence: float = Field(default=0.85, description="Confidence in answer 0-1")


class DecisionExplanation(BaseModel):
    """Explanation of a decision"""

    decision: str = Field(..., description="The decision being explained")
    explanation: str = Field(..., description="Detailed explanation")
    reasoning_factors: List[str] = Field(default_factory=list, description="Key factors")
    supporting_evidence: List[Citation] = Field(
        default_factory=list, description="Supporting evidence citations"
    )


class DocumentMetadata(BaseModel):
    """Metadata about a document"""

    file_path: Optional[str] = Field(None, description="Original file path")
    total_pages: Optional[int] = Field(None, description="Total pages if PDF")
    chunk_count: int = Field(default=0, description="Number of chunks")
    embedding_model: Optional[str] = Field(None, description="Embedding model used")
    ai_provider: str = Field(default="gemini", description="AI provider used")


class AnalysisResult(BaseModel):
    """Complete analysis result for a document"""

    document_type: DocumentType
    document_length: int = Field(..., description="Length of document in characters")
    summary: str = Field(..., description="Summary of the document")
    key_points: List[KeyPoint] = Field(default_factory=list, description="Extracted key points")
    sentiment: Optional[SentimentAnalysis] = Field(None, description="Sentiment analysis")
    entities: List[str] = Field(default_factory=list, description="Named entities found")
    language: str = Field(default="en", description="Detected language")
    processing_time: float = Field(..., description="Time taken to process in seconds")
    confidence_score: float = Field(..., description="Overall analysis confidence 0-1")
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations based on analysis"
    )
    metadata: Optional[DocumentMetadata] = Field(None, description="Document metadata")
    q_and_a_results: List[QuestionAnswer] = Field(
        default_factory=list, description="Question-answer pairs with citations"
    )


class AnalysisRequest(BaseModel):
    """Request model for document analysis"""

    content: str = Field(..., description="Document content to analyze")
    document_type: Optional[DocumentType] = Field(None, description="Document type hint")
    file_path: Optional[str] = Field(None, description="File path for large documents")
    analyze_sentiment: bool = Field(default=True, description="Whether to analyze sentiment")
    max_summary_length: int = Field(default=500, description="Maximum summary length in characters")
    include_recommendations: bool = Field(
        default=True, description="Whether to include recommendations"
    )
    use_embeddings: bool = Field(
        default=False, description="Whether to use embeddings for semantic search"
    )
    ai_provider: str = Field(
        default="gemini", description="AI provider to use: 'gemini' or 'openai'"
    )
