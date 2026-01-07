"""PDF and document processing utilities"""

import io
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Process PDF documents and extract text"""

    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """
        Extract text from PDF file

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text from PDF
        """
        try:
            import PyPDF2

            text_content = []
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text:
                        text_content.append(f"--- Page {page_num + 1} ---\n{text}")

            return "\n".join(text_content)
        except ImportError:
            logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
            raise
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Chunk text into overlapping segments for embedding

        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            overlap: Number of characters to overlap between chunks

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap if end < len(text) else len(text)

        return chunks

    @staticmethod
    def load_document(file_path: str) -> tuple[str, str]:
        """
        Load document and detect type

        Args:
            file_path: Path to document file

        Returns:
            Tuple of (content, document_type)
        """
        path = Path(file_path)

        if path.suffix.lower() == ".pdf":
            content = PDFProcessor.extract_text_from_pdf(file_path)
            return content, "pdf"
        elif path.suffix.lower() in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return content, path.suffix[1:].lower()
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return content, "text"
