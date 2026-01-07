"""
Tests for LangChain integration module.
Tests RAG chains, vector stores, and Q&A capabilities.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.langchain_integration import (
    LangChainDocumentAnalyzer,
    LangChainConfig,
    DocumentAnalysisOutput,
    DecisionExplanationOutput,
)
from src.hybrid_analyzer import HybridDocumentAnalyzer


class TestLangChainConfig:
    """Test LangChain configuration"""

    def test_config_defaults(self):
        """Test default configuration values"""
        assert LangChainConfig.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
        assert LangChainConfig.CHUNK_SIZE == 1000
        assert LangChainConfig.CHUNK_OVERLAP == 100
        assert LangChainConfig.RETRIEVER_K == 5
        assert LangChainConfig.TEMPERATURE == 0.7


class TestLangChainDocumentAnalyzer:
    """Test LangChain document analyzer"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing"""
        with patch("src.langchain_integration.get_config") as mock_config:
            mock_config_inst = MagicMock()
            mock_config_inst.get_gemini_key.return_value = "test-key"
            mock_config_inst.get_openai_key.return_value = "test-key"
            mock_config.return_value = mock_config_inst

            analyzer = LangChainDocumentAnalyzer(llm_provider="gemini")
            return analyzer

    def test_init_gemini_provider(self):
        """Test initialization with Gemini provider"""
        with patch("src.langchain_integration.get_config") as mock_config:
            mock_config_inst = MagicMock()
            mock_config_inst.get_gemini_key.return_value = "test-key"
            mock_config.return_value = mock_config_inst

            analyzer = LangChainDocumentAnalyzer(llm_provider="gemini")
            assert analyzer.llm_provider == "gemini"
            assert analyzer.llm is not None

    def test_init_openai_provider(self):
        """Test initialization with OpenAI provider"""
        with patch("src.langchain_integration.get_config") as mock_config:
            mock_config_inst = MagicMock()
            mock_config_inst.get_openai_key.return_value = "test-key"
            mock_config.return_value = mock_config_inst

            analyzer = LangChainDocumentAnalyzer(llm_provider="openai")
            assert analyzer.llm_provider == "openai"
            assert analyzer.llm is not None

    def test_invalid_llm_provider(self):
        """Test initialization with invalid LLM provider"""
        with patch("src.langchain_integration.get_config") as mock_config:
            mock_config_inst = MagicMock()
            mock_config.return_value = mock_config_inst

            with pytest.raises(ValueError):
                LangChainDocumentAnalyzer(llm_provider="invalid")

    def test_invalid_embedding_provider(self):
        """Test initialization with invalid embedding provider"""
        with patch("src.langchain_integration.get_config") as mock_config:
            mock_config_inst = MagicMock()
            mock_config_inst.get_gemini_key.return_value = "test-key"
            mock_config.return_value = mock_config_inst

            with pytest.raises(ValueError):
                LangChainDocumentAnalyzer(
                    llm_provider="gemini",
                    embedding_provider="invalid",
                )

    def test_embeddings_init_huggingface(self):
        """Test HuggingFace embeddings initialization"""
        with patch("src.langchain_integration.get_config") as mock_config:
            mock_config_inst = MagicMock()
            mock_config_inst.get_gemini_key.return_value = "test-key"
            mock_config.return_value = mock_config_inst

            analyzer = LangChainDocumentAnalyzer(
                llm_provider="gemini",
                embedding_provider="huggingface",
            )
            assert analyzer.embedding_provider == "huggingface"
            assert analyzer.embeddings is not None

    def test_embeddings_init_openai(self):
        """Test OpenAI embeddings initialization"""
        with patch("src.langchain_integration.get_config") as mock_config:
            mock_config_inst = MagicMock()
            mock_config_inst.get_gemini_key.return_value = "test-key"
            mock_config_inst.get_openai_key.return_value = "test-key"
            mock_config.return_value = mock_config_inst

            analyzer = LangChainDocumentAnalyzer(
                llm_provider="gemini",
                embedding_provider="openai",
            )
            assert analyzer.embedding_provider == "openai"

    def test_temperature_setting(self):
        """Test custom temperature setting"""
        with patch("src.langchain_integration.get_config") as mock_config:
            mock_config_inst = MagicMock()
            mock_config_inst.get_gemini_key.return_value = "test-key"
            mock_config.return_value = mock_config_inst

            analyzer = LangChainDocumentAnalyzer(temperature=0.3)
            assert analyzer.temperature == 0.3

    def test_conversation_memory_init(self):
        """Test conversation memory initialization"""
        with patch("src.langchain_integration.get_config") as mock_config:
            mock_config_inst = MagicMock()
            mock_config_inst.get_gemini_key.return_value = "test-key"
            mock_config.return_value = mock_config_inst

            analyzer = LangChainDocumentAnalyzer(llm_provider="gemini")
            assert analyzer.conversation_memory is not None


class TestHybridDocumentAnalyzer:
    """Test hybrid analyzer combining traditional and LangChain approaches"""

    @pytest.fixture
    def hybrid_analyzer(self):
        """Create hybrid analyzer for testing"""
        with patch("src.langchain_integration.get_config") as mock_config:
            mock_config_inst = MagicMock()
            mock_config_inst.get_gemini_key.return_value = "test-key"
            mock_config.return_value = mock_config_inst

            with patch("src.enhanced_analyzer.GeminiClient"):
                analyzer = HybridDocumentAnalyzer(
                    use_langchain=True,
                    llm_provider="gemini",
                )
                return analyzer

    def test_hybrid_init_both_analyzers(self, hybrid_analyzer):
        """Test initialization of both traditional and LangChain analyzers"""
        assert hybrid_analyzer.traditional_analyzer is not None
        assert hybrid_analyzer.langchain_analyzer is not None
        assert hybrid_analyzer.use_langchain is True

    def test_hybrid_init_traditional_only(self):
        """Test initialization with LangChain disabled"""
        with patch("src.langchain_integration.get_config") as mock_config:
            mock_config_inst = MagicMock()
            mock_config.return_value = mock_config_inst

            with patch("src.enhanced_analyzer.GeminiClient"):
                analyzer = HybridDocumentAnalyzer(use_langchain=False)
                assert analyzer.traditional_analyzer is not None
                assert analyzer.langchain_analyzer is None

    def test_hybrid_provides_backward_compatibility(self, hybrid_analyzer):
        """Test that hybrid analyzer maintains backward compatibility"""
        # Should have both traditional and LangChain methods
        assert hasattr(hybrid_analyzer, "analyze")
        assert hasattr(hybrid_analyzer, "answer_question")
        assert hasattr(hybrid_analyzer, "explain_decision")
        assert hasattr(hybrid_analyzer, "conversational_qa")

    def test_clear_conversation(self, hybrid_analyzer):
        """Test clearing conversation memory"""
        hybrid_analyzer.clear_conversation()
        # Should not raise exception

    def test_cleanup(self, hybrid_analyzer):
        """Test cleanup of resources"""
        hybrid_analyzer.cleanup()
        # Should not raise exception


class TestStructuredOutputModels:
    """Test Pydantic models for structured output"""

    def test_document_analysis_output_model(self):
        """Test DocumentAnalysisOutput model"""
        output = DocumentAnalysisOutput(
            summary="Test summary",
            key_points=["Point 1", "Point 2"],
            sentiment="positive",
            entity_types={"PERSON": ["John"], "ORG": ["Company"]},
            recommendations=["Rec 1", "Rec 2"],
        )

        assert output.summary == "Test summary"
        assert len(output.key_points) == 2
        assert output.sentiment == "positive"
        assert "PERSON" in output.entity_types

    def test_decision_explanation_output_model(self):
        """Test DecisionExplanationOutput model"""
        output = DecisionExplanationOutput(
            decision="Approve proposal",
            reasoning=["Reason 1", "Reason 2"],
            supporting_facts=["Fact 1", "Fact 2"],
            confidence_score=0.85,
        )

        assert output.decision == "Approve proposal"
        assert len(output.reasoning) == 2
        assert output.confidence_score == 0.85

    def test_decision_explanation_output_invalid_confidence(self):
        """Test DecisionExplanationOutput with invalid confidence"""
        with pytest.raises(ValueError):
            DecisionExplanationOutput(
                decision="Test",
                reasoning=[],
                supporting_facts=[],
                confidence_score=1.5,  # Invalid: > 1.0
            )


class TestLangChainIntegrationFlow:
    """Test end-to-end LangChain integration flow"""

    def test_analyzer_initialization_flow(self):
        """Test full initialization flow"""
        with patch("src.langchain_integration.get_config") as mock_config:
            mock_config_inst = MagicMock()
            mock_config_inst.get_gemini_key.return_value = "test-key"
            mock_config.return_value = mock_config_inst

            # Should initialize without errors
            analyzer = LangChainDocumentAnalyzer(
                llm_provider="gemini",
                embedding_provider="huggingface",
                vector_store_type="chroma",
            )

            assert analyzer.llm is not None
            assert analyzer.embeddings is not None
            assert analyzer.vector_store is None  # Not created until documents loaded

    def test_hybrid_analyzer_feature_availability(self):
        """Test that hybrid analyzer has expected features"""
        with patch("src.langchain_integration.get_config") as mock_config:
            mock_config_inst = MagicMock()
            mock_config_inst.get_gemini_key.return_value = "test-key"
            mock_config.return_value = mock_config_inst

            with patch("src.enhanced_analyzer.GeminiClient"):
                analyzer = HybridDocumentAnalyzer(use_langchain=True)

                # Should have all required methods
                expected_methods = [
                    "load_document",
                    "answer_question",
                    "conversational_qa",
                    "analyze",
                    "explain_decision",
                    "clear_conversation",
                    "cleanup",
                ]

                for method in expected_methods:
                    assert hasattr(analyzer, method)
                    assert callable(getattr(analyzer, method))
