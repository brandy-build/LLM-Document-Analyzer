"""Google Gemini AI integration"""

import os
import logging
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class GeminiClient:
    """Client for Google Gemini API"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-pro"):
        """
        Initialize Gemini client

        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            model: Model to use (default: gemini-pro)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY must be provided or set in environment")

        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self.client = genai
        except ImportError:
            logger.error("google-generativeai not installed. Install with: pip install google-generativeai")
            raise

    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """
        Generate text using Gemini

        Args:
            prompt: Input prompt
            temperature: Temperature for responses (0-1)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        try:
            model = self.client.GenerativeModel(self.model)
            response = model.generate_content(
                prompt,
                generation_config=self.client.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return response.text
        except Exception as e:
            logger.error(f"Error generating text with Gemini: {e}")
            raise

    def answer_question(
        self,
        question: str,
        context: str,
        temperature: float = 0.7,
    ) -> str:
        """
        Answer question based on context

        Args:
            question: Question to answer
            context: Context/document to use
            temperature: Temperature for responses

        Returns:
            Answer to the question
        """
        prompt = f"""Based on the following context, answer the question. 
If the answer is not in the context, say "I cannot find this information in the provided context."

Context:
{context}

Question: {question}

Answer:"""
        return self.generate_text(prompt, temperature=temperature)

    def explain_decision(
        self,
        decision: str,
        context: str,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Explain why a decision was made

        Args:
            decision: The decision to explain
            context: Context/document related to decision
            temperature: Temperature for responses

        Returns:
            Dictionary with explanation and reasoning
        """
        prompt = f"""Analyze and explain the following decision based on the context provided.
Provide:
1. A clear explanation of the decision
2. Key factors that led to this decision
3. Evidence from the context that supports this decision
4. Any assumptions made

Context:
{context}

Decision: {decision}

Explanation:"""

        explanation_text = self.generate_text(prompt, temperature=temperature)

        return {
            "decision": decision,
            "explanation": explanation_text,
            "reasoning_model": self.model,
        }

    def summarize_with_key_points(
        self,
        text: str,
        max_points: int = 5,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Summarize text and extract key points

        Args:
            text: Text to summarize
            max_points: Maximum key points to extract
            temperature: Temperature for responses

        Returns:
            Dictionary with summary and key points
        """
        prompt = f"""Summarize the following text and extract {max_points} key points.
Format your response as JSON with 'summary' and 'key_points' fields.

Text:
{text}

JSON Response:"""

        try:
            import json

            response = self.generate_text(prompt, temperature=temperature)
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            logger.warning("Could not parse Gemini response as JSON, returning raw response")
            return {"summary": response, "key_points": []}

    def extract_entities(self, text: str, entity_types: Optional[List[str]] = None) -> List[Dict]:
        """
        Extract named entities from text

        Args:
            text: Text to analyze
            entity_types: Types of entities to extract (optional)

        Returns:
            List of extracted entities
        """
        entity_desc = (
            f"Types: {', '.join(entity_types)}" if entity_types else "Common types: person, organization, location"
        )

        prompt = f"""Extract named entities from the following text.
{entity_desc}

Return as JSON array with 'text' and 'type' fields.

Text:
{text}

JSON Response:"""

        try:
            import json

            response = self.generate_text(prompt)
            entities = json.loads(response)
            return entities if isinstance(entities, list) else []
        except json.JSONDecodeError:
            logger.warning("Could not parse entities as JSON")
            return []
