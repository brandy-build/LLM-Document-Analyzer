"""Main document analyzer using LLM"""

import os
import time
import logging
from typing import Optional, List
from dotenv import load_dotenv
from openai import OpenAI

from .models import (
    AnalysisRequest,
    AnalysisResult,
    DocumentType,
    SentimentAnalysis,
    KeyPoint,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


class DocumentAnalyzer:
    """Analyzes documents using OpenAI's LLM"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
    ):
        """
        Initialize the document analyzer

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (default: gpt-4-turbo-preview)
            temperature: Temperature for model responses (0-1)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", model)
        self.temperature = temperature or float(os.getenv("TEMPERATURE", temperature))

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be provided or set in environment")

        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"DocumentAnalyzer initialized with model: {self.model}")

    def detect_document_type(self, content: str) -> DocumentType:
        """Detect the type of document"""
        try:
            # Simple heuristics for document type detection
            if content.strip().startswith("%PDF"):
                return DocumentType.PDF
            elif content.strip().startswith("#"):
                return DocumentType.MARKDOWN
            elif any(
                content.startswith(keyword) for keyword in ["def ", "class ", "import ", "from "]
            ):
                return DocumentType.CODE
            elif "Subject:" in content and "From:" in content:
                return DocumentType.EMAIL
            else:
                return DocumentType.TEXT
        except Exception as e:
            logger.warning(f"Error detecting document type: {e}")
            return DocumentType.UNKNOWN

    def _call_llm(self, prompt: str) -> str:
        """Call OpenAI LLM with the given prompt"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise

    def _extract_summary(self, content: str, max_length: int = 500) -> str:
        """Extract a summary of the document"""
        prompt = f"""Provide a concise summary of the following document in at most {max_length} characters:

{content[:2000]}

Summary:"""
        return self._call_llm(prompt).strip()

    def _extract_key_points(self, content: str) -> List[KeyPoint]:
        """Extract key points from the document"""
        prompt = f"""Extract the 5 most important key points from this document. 
For each point, rate its importance from 0 to 1.
Format your response as JSON array with objects containing 'text' and 'importance' fields.

Document:
{content[:2000]}

Key Points (JSON):"""

        try:
            response = self._call_llm(prompt)
            # Parse JSON response
            import json

            points_data = json.loads(response)
            return [
                KeyPoint(text=point["text"], importance=float(point["importance"]))
                for point in points_data
            ]
        except Exception as e:
            logger.warning(f"Error extracting key points: {e}")
            return []

    def _analyze_sentiment(self, content: str) -> Optional[SentimentAnalysis]:
        """Analyze sentiment of the document"""
        prompt = f"""Analyze the sentiment of this document. Respond in JSON format with:
- overall_sentiment: one of 'positive', 'negative', 'neutral'
- confidence: number between 0 and 1
- emotional_tone: brief description of emotional tone

Document:
{content[:2000]}

Analysis (JSON):"""

        try:
            response = self._call_llm(prompt)
            import json

            sentiment_data = json.loads(response)
            return SentimentAnalysis(**sentiment_data)
        except Exception as e:
            logger.warning(f"Error analyzing sentiment: {e}")
            return None

    def _extract_entities(self, content: str) -> List[str]:
        """Extract named entities from the document"""
        prompt = f"""Extract the main named entities (people, organizations, locations, products) from this document.
Return them as a JSON array of strings.

Document:
{content[:2000]}

Entities (JSON array):"""

        try:
            response = self._call_llm(prompt)
            import json

            entities = json.loads(response)
            return entities if isinstance(entities, list) else []
        except Exception as e:
            logger.warning(f"Error extracting entities: {e}")
            return []

    def _generate_recommendations(self, content: str, summary: str) -> List[str]:
        """Generate recommendations based on document analysis"""
        prompt = f"""Based on this document, provide 3-5 actionable recommendations:

{content[:1500]}

Recommendations (as JSON array of strings):"""

        try:
            response = self._call_llm(prompt)
            import json

            recommendations = json.loads(response)
            return recommendations if isinstance(recommendations, list) else []
        except Exception as e:
            logger.warning(f"Error generating recommendations: {e}")
            return []

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Analyze a document and return comprehensive analysis results

        Args:
            request: AnalysisRequest containing document content and options

        Returns:
            AnalysisResult with all analysis outputs
        """
        start_time = time.time()
        logger.info("Starting document analysis...")

        content = request.content
        document_type = request.document_type or self.detect_document_type(content)

        # Extract summary
        summary = self._extract_summary(content, max_length=request.max_summary_length)
        logger.info("Summary extracted")

        # Extract key points
        key_points = self._extract_key_points(content)
        logger.info(f"Extracted {len(key_points)} key points")

        # Analyze sentiment
        sentiment = None
        if request.analyze_sentiment:
            sentiment = self._analyze_sentiment(content)
            logger.info("Sentiment analysis completed")

        # Extract entities
        entities = self._extract_entities(content)
        logger.info(f"Extracted {len(entities)} entities")

        # Generate recommendations
        recommendations = []
        if request.include_recommendations:
            recommendations = self._generate_recommendations(content, summary)
            logger.info(f"Generated {len(recommendations)} recommendations")

        processing_time = time.time() - start_time

        result = AnalysisResult(
            document_type=document_type,
            document_length=len(content),
            summary=summary,
            key_points=key_points,
            sentiment=sentiment,
            entities=entities,
            processing_time=processing_time,
            confidence_score=0.85,  # Default confidence
            recommendations=recommendations,
        )

        logger.info(f"Analysis completed in {processing_time:.2f} seconds")
        return result
