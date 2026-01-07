"""Embedding and vector store utilities"""

import logging
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """Store and retrieve document embeddings for semantic search"""

    def __init__(self):
        """Initialize embedding store"""
        self.embeddings: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}

    def add_embedding(
        self, text: str, embedding: List[float], chunk_id: int, metadata: Optional[Dict] = None
    ) -> None:
        """
        Add embedding to store

        Args:
            text: Original text
            embedding: Embedding vector
            chunk_id: Chunk identifier
            metadata: Additional metadata
        """
        entry = {
            "chunk_id": chunk_id,
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {},
        }
        self.embeddings.append(entry)

    def semantic_search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """
        Find most similar embeddings to query

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of most similar chunks
        """
        import math

        def cosine_similarity(a: List[float], b: List[float]) -> float:
            """Calculate cosine similarity between vectors"""
            dot_product = sum(x * y for x, y in zip(a, b))
            magnitude_a = math.sqrt(sum(x * x for x in a))
            magnitude_b = math.sqrt(sum(x * x for x in b))
            if magnitude_a == 0 or magnitude_b == 0:
                return 0
            return dot_product / (magnitude_a * magnitude_b)

        similarities = []
        for entry in self.embeddings:
            score = cosine_similarity(query_embedding, entry["embedding"])
            similarities.append((entry, score))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in similarities[:top_k]]

    def save(self, file_path: str) -> None:
        """Save embeddings to file"""
        with open(file_path, "w") as f:
            json.dump(
                {
                    "embeddings": self.embeddings,
                    "metadata": self.metadata,
                },
                f,
            )
        logger.info(f"Embeddings saved to {file_path}")

    def load(self, file_path: str) -> None:
        """Load embeddings from file"""
        with open(file_path, "r") as f:
            data = json.load(f)
            self.embeddings = data["embeddings"]
            self.metadata = data["metadata"]
        logger.info(f"Embeddings loaded from {file_path}")


class Citation:
    """Represents a citation with source location"""

    def __init__(
        self, text: str, chunk_id: int, page: Optional[int] = None, confidence: float = 1.0
    ):
        """
        Initialize citation

        Args:
            text: Source text
            chunk_id: Chunk identifier
            page: Page number if applicable
            confidence: Confidence score
        """
        self.text = text
        self.chunk_id = chunk_id
        self.page = page
        self.confidence = confidence

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "page": self.page,
            "confidence": self.confidence,
        }

    def __repr__(self) -> str:
        page_info = f", page {self.page}" if self.page else ""
        return f'"{self.text}"... (chunk {self.chunk_id}{page_info}, confidence: {self.confidence:.2f})'
