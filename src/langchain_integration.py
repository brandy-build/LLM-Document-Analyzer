"""
LangChain Integration Module
Provides LangChain-based RAG pipeline for document analysis and Q&A.
Supports Gemini and OpenAI as LLM providers with Chroma as vector store.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import tempfile
import shutil

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_community.vectorstores import Chroma
    from langchain.chains import RetrievalQA, ConversationalRetrievalChain
    from langchain.chains.llm import LLMChain
    from langchain.prompts import PromptTemplate, ChatPromptTemplate
    from langchain.memory import ConversationBufferMemory
    from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
    from langchain_core.callbacks import CallbackManager
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError as e:
    raise ImportError(f"LangChain components not installed. Install with: pip install -e '.[langchain]' - {e}")

from pydantic import BaseModel, Field

from .secure_config import get_config

logger = logging.getLogger(__name__)


class LangChainConfig:
    """Configuration for LangChain components"""

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    RETRIEVER_K = 5  # Number of top documents to retrieve
    TEMPERATURE = 0.7


class DocumentAnalysisOutput(BaseModel):
    """Structured output for document analysis"""

    summary: str = Field(description="Summary of the document")
    key_points: List[str] = Field(description="List of key points")
    sentiment: str = Field(description="Overall sentiment (positive/negative/neutral)")
    entity_types: Dict[str, List[str]] = Field(description="Named entities by type")
    recommendations: List[str] = Field(description="Actionable recommendations")


class DecisionExplanationOutput(BaseModel):
    """Structured output for decision explanation"""

    decision: str = Field(description="The decision made")
    reasoning: List[str] = Field(description="Step-by-step reasoning")
    supporting_facts: List[str] = Field(description="Facts supporting the decision")
    confidence_score: float = Field(description="Confidence score (0-1)")


class CitationAnswer(BaseModel):
    """Structured output for Q&A with citations"""

    answer: str = Field(description="The answer to the question")
    citations: List[Dict[str, Any]] = Field(description="Source citations")
    confidence_score: float = Field(description="Answer confidence (0-1)")


class LangChainDocumentAnalyzer:
    """
    LangChain-based document analyzer supporting RAG and Q&A with memory.
    Supports multiple LLM providers and vector stores.
    """

    def __init__(
        self,
        llm_provider: str = "gemini",
        embedding_provider: str = "huggingface",
        vector_store_type: str = "chroma",
        persist_directory: Optional[str] = None,
        temperature: float = LangChainConfig.TEMPERATURE,
    ):
        """
        Initialize LangChain analyzer.

        Args:
            llm_provider: LLM to use ('gemini' or 'openai')
            embedding_provider: Embedding provider ('huggingface' or 'openai')
            vector_store_type: Vector store type ('chroma' or 'faiss')
            persist_directory: Directory to persist vector store
            temperature: Temperature for LLM responses
        """
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.vector_store_type = vector_store_type
        self.persist_directory = persist_directory or tempfile.mkdtemp()
        self.temperature = temperature

        # Initialize components
        self.llm = self._init_llm()
        self.embeddings = self._init_embeddings()
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        self.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self.conversational_chain = None

        logger.info(
            f"LangChain analyzer initialized with {llm_provider} LLM, "
            f"{embedding_provider} embeddings, {vector_store_type} vector store"
        )

    def _init_llm(self):
        """Initialize LLM based on provider"""
        config = get_config()

        if self.llm_provider == "gemini":
            gemini_key = config.get_gemini_key()
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=gemini_key,
                temperature=self.temperature,
                top_p=0.95,
            )
        elif self.llm_provider == "openai":
            openai_key = config.get_openai_key()
            return ChatOpenAI(
                model="gpt-4-turbo-preview",
                api_key=openai_key,
                temperature=self.temperature,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _init_embeddings(self):
        """Initialize embeddings based on provider"""
        if self.embedding_provider == "huggingface":
            return HuggingFaceEmbeddings(model_name=LangChainConfig.EMBEDDING_MODEL)
        elif self.embedding_provider == "openai":
            config = get_config()
            openai_key = config.get_openai_key()
            return OpenAIEmbeddings(api_key=openai_key)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")

    def load_document(self, file_path: str) -> List[Any]:
        """
        Load document using LangChain document loaders.

        Args:
            file_path: Path to document (PDF, TXT, etc.)

        Returns:
            List of LangChain Document objects
        """
        file_path = Path(file_path)

        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix.lower() in [".txt", ".md"]:
            loader = TextLoader(str(file_path))
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from {file_path}")
        return documents

    def process_documents(self, documents: List[Any]) -> List[Any]:
        """
        Split documents into chunks using RecursiveCharacterTextSplitter.

        Args:
            documents: List of LangChain Document objects

        Returns:
            List of chunked documents
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=LangChainConfig.CHUNK_SIZE,
            chunk_overlap=LangChainConfig.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )

        split_docs = splitter.split_documents(documents)
        logger.info(f"Split documents into {len(split_docs)} chunks")
        return split_docs

    def create_vector_store(self, documents: List[Any], collection_name: str = "documents"):
        """
        Create and populate vector store with document embeddings.

        Args:
            documents: List of LangChain Document objects
            collection_name: Name for the vector store collection
        """
        if self.vector_store_type == "chroma":
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=collection_name,
            )
            self.vector_store.persist()
        else:
            raise ValueError(f"Unsupported vector store: {self.vector_store_type}")

        # Create retriever from vector store
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": LangChainConfig.RETRIEVER_K}
        )

        logger.info(f"Created {self.vector_store_type} vector store with {len(documents)} documents")

    def setup_qa_chain(self):
        """Setup RetrievalQA chain for question answering"""
        if not self.retriever:
            raise ValueError("Vector store not initialized. Call create_vector_store first.")

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            verbose=True,
        )

        logger.info("RetrievalQA chain initialized")

    def setup_conversational_chain(self):
        """Setup ConversationalRetrievalChain for multi-turn conversation"""
        if not self.retriever:
            raise ValueError("Vector store not initialized. Call create_vector_store first.")

        self.conversational_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.conversation_memory,
            return_source_documents=True,
            verbose=True,
        )

        logger.info("ConversationalRetrievalChain initialized")

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using RetrievalQA chain.

        Args:
            question: Question to answer

        Returns:
            Dict with answer and source documents
        """
        if not self.qa_chain:
            self.setup_qa_chain()

        result = self.qa_chain.invoke({"query": question})

        # Extract citations from source documents
        citations = []
        for doc in result.get("source_documents", []):
            citations.append(
                {
                    "content": doc.page_content,
                    "page": doc.metadata.get("page", 0),
                    "source": doc.metadata.get("source", "unknown"),
                }
            )

        return {
            "answer": result["result"],
            "citations": citations,
            "confidence_score": 0.85,  # Could be calculated from retrieval scores
        }

    def conversational_qa(self, question: str) -> Dict[str, Any]:
        """
        Answer a question with conversation history.

        Args:
            question: Question to answer

        Returns:
            Dict with answer, source documents, and conversation context
        """
        if not self.conversational_chain:
            self.setup_conversational_chain()

        result = self.conversational_chain.invoke({"question": question})

        citations = []
        for doc in result.get("source_documents", []):
            citations.append(
                {
                    "content": doc.page_content,
                    "page": doc.metadata.get("page", 0),
                    "source": doc.metadata.get("source", "unknown"),
                }
            )

        return {
            "answer": result["answer"],
            "citations": citations,
            "chat_history": self.conversation_memory.buffer,
        }

    def analyze_document(self, documents: List[Any]) -> Dict[str, Any]:
        """
        Analyze document using structured output with PydanticOutputParser.

        Args:
            documents: List of LangChain Document objects

        Returns:
            Structured analysis output
        """
        # Combine document content
        doc_content = "\n\n".join([doc.page_content for doc in documents])

        # Create parser
        parser = PydanticOutputParser(pydantic_object=DocumentAnalysisOutput)

        # Create prompt with format instructions
        prompt = ChatPromptTemplate.from_template(
            """Analyze the following document and provide structured output.
            
Document:
{document}

{format_instructions}

Provide your analysis in the requested JSON format."""
        )

        # Create chain
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
        )

        # Run chain
        try:
            result = chain.run(
                document=doc_content,
                format_instructions=parser.get_format_instructions(),
            )

            # Parse output
            parsed_result = parser.parse(result)
            return parsed_result.model_dump()
        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            return {
                "summary": "Error analyzing document",
                "key_points": [],
                "sentiment": "unknown",
                "entity_types": {},
                "recommendations": [],
            }

    def explain_decision(self, context: str, decision: str) -> Dict[str, Any]:
        """
        Explain a decision with reasoning using structured output.

        Args:
            context: Context for the decision
            decision: The decision to explain

        Returns:
            Structured explanation with reasoning
        """
        parser = PydanticOutputParser(pydantic_object=DecisionExplanationOutput)

        prompt = ChatPromptTemplate.from_template(
            """Given the following context and decision, explain the reasoning.

Context:
{context}

Decision:
{decision}

{format_instructions}

Provide a detailed explanation in the requested JSON format."""
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)

        try:
            result = chain.run(
                context=context,
                decision=decision,
                format_instructions=parser.get_format_instructions(),
            )

            parsed_result = parser.parse(result)
            return parsed_result.model_dump()
        except Exception as e:
            logger.error(f"Error explaining decision: {e}")
            return {
                "decision": decision,
                "reasoning": [f"Error: {str(e)}"],
                "supporting_facts": [],
                "confidence_score": 0.0,
            }

    def clear_conversation(self):
        """Clear conversation memory for new conversation"""
        self.conversation_memory.clear()
        logger.info("Conversation memory cleared")

    def cleanup(self):
        """Clean up resources and persist vector store"""
        if self.vector_store:
            self.vector_store.persist()
            logger.info("Vector store persisted")

        if self.persist_directory and Path(self.persist_directory).exists():
            # Only delete if it's a temporary directory
            if self.persist_directory.startswith(tempfile.gettempdir()):
                shutil.rmtree(self.persist_directory)
                logger.info(f"Cleaned up temporary directory: {self.persist_directory}")
