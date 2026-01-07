"""Enhanced document analyzer with embeddings, Q&A, and Gemini support"""

import os
import time
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

from .models import (
    AnalysisRequest,
    AnalysisResult,
    DocumentType,
    SentimentAnalysis,
    KeyPoint,
    QuestionAnswer,
    Citation,
    DocumentMetadata,
)
from .document_processor import PDFProcessor
from .embeddings import EmbeddingStore, Citation as EmbeddingCitation
from .gemini_client import GeminiClient

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


class EnhancedDocumentAnalyzer:
    """Analyzes documents using LLM with embeddings and Q&A support"""

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        default_provider: str = "gemini",
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize the enhanced document analyzer

        Args:
            gemini_api_key: Google Gemini API key (defaults to GOOGLE_API_KEY env var)
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            default_provider: Default AI provider ('gemini' or 'openai')
            embedding_model: Embedding model to use (optional)
        """
        self.gemini_api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.default_provider = default_provider
        self.embedding_model = embedding_model
        self.gemini_client = None
        self.openai_client = None
        self.embedding_store = EmbeddingStore()

        # Initialize Gemini if key is available
        if self.gemini_api_key:
            try:
                self.gemini_client = GeminiClient(api_key=self.gemini_api_key)
                logger.info("Gemini client initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Gemini client: {e}")

        # Initialize OpenAI if key is available
        if self.openai_api_key:
            try:
                from openai import OpenAI

                self.openai_client = OpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}")

        if not self.gemini_client and not self.openai_client:
            raise ValueError(
                "At least one API key (GOOGLE_API_KEY or OPENAI_API_KEY) must be provided"
            )

    def load_document(self, file_path: str) -> str:
        """
        Load document from file (supports PDF, TXT, MD)

        Args:
            file_path: Path to document

        Returns:
            Document content
        """
        logger.info(f"Loading document: {file_path}")
        content, doc_type = PDFProcessor.load_document(file_path)
        logger.info(f"Loaded {len(content)} characters from {doc_type} document")
        return content

    def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for text chunks

        Args:
            chunks: List of text chunks

        Returns:
            List of embedding vectors
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks")

        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.embedding_model or "all-MiniLM-L6-v2")
            embeddings = model.encode(chunks, show_progress_bar=False)
            return embeddings.tolist()
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise

    def build_embedding_index(self, content: str, chunk_size: int = 1000) -> EmbeddingStore:
        """
        Build embedding index for document content

        Args:
            content: Document content
            chunk_size: Size of each chunk

        Returns:
            EmbeddingStore with indexed chunks
        """
        logger.info("Building embedding index...")

        # Chunk the document
        chunks = PDFProcessor.chunk_text(content, chunk_size=chunk_size)
        logger.info(f"Created {len(chunks)} chunks")

        # Generate embeddings
        embeddings = self.generate_embeddings(chunks)

        # Build store
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            self.embedding_store.add_embedding(
                text=chunk,
                embedding=embedding,
                chunk_id=i,
                metadata={"chunk_size": len(chunk)},
            )

        logger.info("Embedding index built successfully")
        return self.embedding_store

    def answer_question(
        self, question: str, use_embeddings: bool = False, provider: Optional[str] = None
    ) -> QuestionAnswer:
        """
        Answer a question based on document context

        Args:
            question: Question to answer
            use_embeddings: Whether to use embeddings for semantic search
            provider: AI provider to use ('gemini' or 'openai')

        Returns:
            QuestionAnswer with citations
        """
        provider = provider or self.default_provider
        logger.info(f"Answering question with {provider}: {question}")

        # Find relevant context using embeddings if available
        context = ""
        citations: List[Citation] = []

        if use_embeddings and self.embedding_store.embeddings:
            try:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(self.embedding_model or "all-MiniLM-L6-v2")
                query_embedding = model.encode(question).tolist()
                relevant_chunks = self.embedding_store.semantic_search(query_embedding, top_k=3)

                for entry in relevant_chunks:
                    context += f"\n{entry['text']}\n"
                    citations.append(
                        Citation(
                            text=entry["text"][:200] + "...",
                            chunk_id=entry["chunk_id"],
                            confidence=0.85,
                        )
                    )
            except Exception as e:
                logger.warning(f"Error using embeddings: {e}")

        # Generate answer
        if provider == "gemini" and self.gemini_client:
            answer_text = self.gemini_client.answer_question(question, context or "Context not available")
        elif provider == "openai" and self.openai_client:
            answer_text = self._answer_with_openai(question, context)
        else:
            raise ValueError(f"Provider {provider} not available")

        return QuestionAnswer(
            question=question,
            answer=answer_text,
            citations=citations,
            confidence=0.85 if citations else 0.7,
        )

    def _answer_with_openai(self, question: str, context: str) -> str:
        """Answer using OpenAI"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
                ],
                temperature=0.7,
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error with OpenAI: {e}")
            raise

    def explain_decision(
        self, decision: str, context: str, provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Explain why a decision was made

        Args:
            decision: The decision to explain
            context: Relevant context
            provider: AI provider to use

        Returns:
            Explanation dictionary
        """
        provider = provider or self.default_provider
        logger.info(f"Explaining decision with {provider}")

        if provider == "gemini" and self.gemini_client:
            return self.gemini_client.explain_decision(decision, context)
        elif provider == "openai" and self.openai_client:
            return self._explain_with_openai(decision, context)
        else:
            raise ValueError(f"Provider {provider} not available")

    def _explain_with_openai(self, decision: str, context: str) -> Dict[str, Any]:
        """Explain decision using OpenAI"""
        try:
            prompt = f"""Explain why this decision was made based on the context.
Provide reasoning and supporting factors.

Context: {context}
Decision: {decision}

Explanation:"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000,
            )

            return {
                "decision": decision,
                "explanation": response.choices[0].message.content,
                "reasoning_model": "gpt-4-turbo-preview",
            }
        except Exception as e:
            logger.error(f"Error with OpenAI: {e}")
            raise

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Comprehensive document analysis

        Args:
            request: AnalysisRequest with document and options

        Returns:
            AnalysisResult with all analyses
        """
        start_time = time.time()
        logger.info("Starting enhanced document analysis...")

        # Load document if file path provided
        content = request.content
        if request.file_path:
            content = self.load_document(request.file_path)

        # Build embedding index if requested
        metadata = DocumentMetadata(
            file_path=request.file_path,
            ai_provider=request.ai_provider,
        )

        if request.use_embeddings:
            self.build_embedding_index(content)
            metadata.chunk_count = len(self.embedding_store.embeddings)
            metadata.embedding_model = self.embedding_model

        # Detect document type
        doc_type = request.document_type or DocumentType.TEXT

        # Use specified provider
        provider = request.ai_provider

        # Generate basic analysis using selected provider
        if provider == "gemini" and self.gemini_client:
            summary = self._summarize_with_gemini(content[:5000], request.max_summary_length)
        elif provider == "openai" and self.openai_client:
            summary = self._summarize_with_openai(content[:5000], request.max_summary_length)
        else:
            summary = content[:request.max_summary_length]

        processing_time = time.time() - start_time

        result = AnalysisResult(
            document_type=doc_type,
            document_length=len(content),
            summary=summary,
            key_points=[],
            processing_time=processing_time,
            confidence_score=0.85,
            metadata=metadata,
        )

        logger.info(f"Analysis completed in {processing_time:.2f} seconds")
        return result

    def _summarize_with_gemini(self, content: str, max_length: int) -> str:
        """Summarize using Gemini"""
        try:
            prompt = f"Summarize the following text in {max_length} characters:\n\n{content}\n\nSummary:"
            return self.gemini_client.generate_text(prompt)
        except Exception as e:
            logger.warning(f"Error with Gemini summarization: {e}")
            return content[:max_length]

    def _summarize_with_openai(self, content: str, max_length: int) -> str:
        """Summarize using OpenAI"""
        try:
            prompt = f"Summarize the following text in {max_length} characters:\n\n{content}\n\nSummary:"
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"Error with OpenAI summarization: {e}")
            return content[:max_length]
