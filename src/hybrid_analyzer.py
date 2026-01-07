"""
Hybrid analyzer combining LangChain RAG with original enhanced analyzer.
Provides backward compatibility while leveraging LangChain capabilities.
"""

import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

from .enhanced_analyzer import EnhancedDocumentAnalyzer
from .models import (
    AnalysisResult,
    QuestionAnswer,
    Citation,
    DocumentMetadata,
)

logger = logging.getLogger(__name__)


class HybridDocumentAnalyzer:
    """
    Hybrid analyzer that combines traditional EnhancedDocumentAnalyzer
    with LangChain RAG capabilities.
    
    Supports both old API (for backward compatibility) and new LangChain API.
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        use_langchain: bool = True,
        llm_provider: str = "gemini",
        embedding_provider: str = "huggingface",
        vector_store_type: str = "chroma",
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize hybrid analyzer.

        Args:
            gemini_api_key: Google Gemini API key
            openai_api_key: OpenAI API key
            use_langchain: Whether to use LangChain components
            llm_provider: LLM provider ('gemini' or 'openai')
            embedding_provider: Embedding provider
            vector_store_type: Vector store type
            persist_directory: Directory to persist vector store
        """
        # Initialize traditional analyzer for backward compatibility
        self.traditional_analyzer = EnhancedDocumentAnalyzer(
            gemini_api_key=gemini_api_key,
            openai_api_key=openai_api_key,
            default_provider=llm_provider,
        )

        # Initialize LangChain analyzer if requested (lazy import)
        self.langchain_analyzer = None
        self.use_langchain = use_langchain
        self._langchain_params = {
            "llm_provider": llm_provider,
            "embedding_provider": embedding_provider,
            "vector_store_type": vector_store_type,
            "persist_directory": persist_directory,
        }

        if use_langchain:
            try:
                # Lazy import to avoid dependency issues
                from .langchain_integration import LangChainDocumentAnalyzer
                
                self.langchain_analyzer = LangChainDocumentAnalyzer(
                    llm_provider=llm_provider,
                    embedding_provider=embedding_provider,
                    vector_store_type=vector_store_type,
                    persist_directory=persist_directory,
                )
            except ImportError as e:
                logger.warning(f"LangChain components not available: {e}")
                self.use_langchain = False

        self.current_documents = []
        logger.info(
            f"Hybrid analyzer initialized with LangChain={self.use_langchain}, "
            f"provider={llm_provider}"
        )

    def load_document(self, file_path: str, use_langchain: bool = None) -> Dict[str, Any]:
        """
        Load and process document.

        Args:
            file_path: Path to document
            use_langchain: Override use_langchain setting for this call

        Returns:
            Document metadata and processing info
        """
        use_lc = use_langchain if use_langchain is not None else self.use_langchain

        if use_lc and self.langchain_analyzer:
            # Load with LangChain
            documents = self.langchain_analyzer.load_document(file_path)
            processed_docs = self.langchain_analyzer.process_documents(documents)
            self.current_documents = processed_docs
            self.langchain_analyzer.create_vector_store(processed_docs)

            return {
                "status": "success",
                "file_path": file_path,
                "document_count": len(processed_docs),
                "method": "langchain",
            }
        else:
            # Load with traditional analyzer
            processor = self.traditional_analyzer.processor
            doc_content, doc_type = processor.load_document(file_path)
            self.traditional_analyzer.current_content = doc_content

            return {
                "status": "success",
                "file_path": file_path,
                "document_type": doc_type,
                "content_length": len(doc_content),
                "method": "traditional",
            }

    def answer_question(self, question: str, use_citations: bool = True) -> QuestionAnswer:
        """
        Answer a question about loaded document.

        Args:
            question: Question to answer
            use_citations: Whether to include source citations

        Returns:
            QuestionAnswer with answer and citations
        """
        if self.use_langchain and self.langchain_analyzer:
            # Use LangChain for Q&A with automatic citations
            try:
                result = self.langchain_analyzer.answer_question(question)

                citations = [
                    Citation(
                        text=c["content"],
                        page=c["page"],
                        source=c["source"],
                        confidence=c.get("confidence", 0.85),
                    )
                    for c in result["citations"]
                ]

                return QuestionAnswer(
                    question=question,
                    answer=result["answer"],
                    confidence=result["confidence_score"],
                    citations=citations if use_citations else [],
                )
            except Exception as e:
                logger.error(f"LangChain Q&A failed: {e}")
                return QuestionAnswer(
                    question=question,
                    answer=f"Error: {str(e)}",
                    confidence=0.0,
                    citations=[],
                )
        else:
            # Fall back to traditional analyzer
            return self.traditional_analyzer.answer_question(
                question, use_embeddings=True
            )

    def conversational_qa(self, question: str) -> Dict[str, Any]:
        """
        Answer question with conversation history (LangChain only).

        Args:
            question: Question to answer

        Returns:
            Dict with answer, citations, and chat history
        """
        if not self.langchain_analyzer:
            raise ValueError("LangChain analyzer not initialized")

        try:
            result = self.langchain_analyzer.conversational_qa(question)
            return result
        except Exception as e:
            logger.error(f"Conversational Q&A failed: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "citations": [],
                "chat_history": [],
            }

    def analyze(self, provider: str = "gemini") -> AnalysisResult:
        """
        Perform full document analysis.

        Args:
            provider: LLM provider to use

        Returns:
            Full analysis result
        """
        return self.traditional_analyzer.analyze(provider=provider)

    def explain_decision(self, decision: str, context: str = "") -> Dict[str, Any]:
        """
        Explain a decision with reasoning.

        Args:
            decision: Decision to explain
            context: Context for the decision

        Returns:
            Explanation with reasoning
        """
        if self.langchain_analyzer:
            return self.langchain_analyzer.explain_decision(context, decision)
        else:
            return self.traditional_analyzer.explain_decision(decision)

    def clear_conversation(self):
        """Clear conversation memory"""
        if self.langchain_analyzer:
            self.langchain_analyzer.clear_conversation()

    def cleanup(self):
        """Clean up resources"""
        if self.langchain_analyzer:
            self.langchain_analyzer.cleanup()
