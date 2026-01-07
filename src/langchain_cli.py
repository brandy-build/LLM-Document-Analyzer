"""
LangChain CLI - Command-line interface for LangChain RAG features.
Provides commands for document loading, Q&A with conversation history, and analysis.
"""

import click
import logging
from pathlib import Path
from typing import Optional

from .hybrid_analyzer import HybridDocumentAnalyzer
from .secure_config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="2.1.0", prog_name="LangChain Analyzer")
def langchain_cli():
    """LangChain-powered document analyzer with RAG capabilities."""
    pass


@langchain_cli.command()
@click.argument("document", type=click.Path(exists=True))
@click.option(
    "--provider",
    default="gemini",
    type=click.Choice(["gemini", "openai"]),
    help="LLM provider to use",
)
@click.option(
    "--embedding-provider",
    default="huggingface",
    type=click.Choice(["huggingface", "openai"]),
    help="Embedding provider",
)
@click.option(
    "--vector-store",
    default="chroma",
    type=click.Choice(["chroma"]),
    help="Vector store type",
)
@click.option(
    "--persist-dir",
    default=None,
    help="Directory to persist vector store",
)
def load(
    document: str,
    provider: str,
    embedding_provider: str,
    vector_store: str,
    persist_dir: Optional[str],
):
    """Load and process a document for Q&A."""
    try:
        click.echo(f"Loading document: {document}")

        analyzer = HybridDocumentAnalyzer(
            use_langchain=True,
            llm_provider=provider,
            embedding_provider=embedding_provider,
            vector_store_type=vector_store,
            persist_directory=persist_dir,
        )

        result = analyzer.load_document(document)

        click.echo(f"✓ Document loaded successfully")
        click.echo(f"  Method: {result['method']}")
        click.echo(f"  Documents/Chunks: {result.get('document_count', 'N/A')}")
        click.echo(f"\nYou can now ask questions about the document.")
        click.echo(f"Use: langchain-cli ask '<question>'")

    except Exception as e:
        click.echo(f"✗ Error loading document: {e}", err=True)
        raise click.Abort()


@langchain_cli.command()
@click.argument("question")
@click.option(
    "--provider",
    default="gemini",
    type=click.Choice(["gemini", "openai"]),
    help="LLM provider to use",
)
@click.option(
    "--with-citations",
    is_flag=True,
    default=True,
    help="Include source citations",
)
def ask(question: str, provider: str, with_citations: bool):
    """Ask a question about the loaded document."""
    try:
        analyzer = HybridDocumentAnalyzer(
            use_langchain=True,
            llm_provider=provider,
        )

        click.echo(f"Analyzing question: {question}\n")

        result = analyzer.answer_question(question, use_citations=with_citations)

        click.echo(f"Answer: {result.answer}")
        click.echo(f"Confidence: {result.confidence:.2%}")

        if result.citations and with_citations:
            click.echo(f"\nSources:")
            for i, citation in enumerate(result.citations, 1):
                click.echo(f"  {i}. Page {citation.page} - {citation.source}")
                click.echo(f"     {citation.text[:100]}...")

    except Exception as e:
        click.echo(f"✗ Error answering question: {e}", err=True)
        raise click.Abort()


@langchain_cli.command()
@click.argument("question")
@click.option(
    "--provider",
    default="gemini",
    type=click.Choice(["gemini", "openai"]),
    help="LLM provider to use",
)
def converse(question: str, provider: str):
    """Multi-turn conversation with the document (with history)."""
    try:
        analyzer = HybridDocumentAnalyzer(
            use_langchain=True,
            llm_provider=provider,
        )

        click.echo(f"Starting multi-turn conversation...")
        click.echo(f"Type 'exit' or 'quit' to end conversation.\n")

        while True:
            user_input = question if question else click.prompt("You")

            if user_input.lower() in ["exit", "quit"]:
                click.echo("Goodbye!")
                analyzer.clear_conversation()
                break

            try:
                result = analyzer.conversational_qa(user_input)

                click.echo(f"\nAssistant: {result['answer']}")

                if result.get("citations"):
                    click.echo(f"Citations: {len(result['citations'])} sources")

                question = None  # Clear question after first iteration
                click.echo()

            except Exception as e:
                click.echo(f"Error: {e}\n")
                question = None

    except Exception as e:
        click.echo(f"✗ Error starting conversation: {e}", err=True)
        raise click.Abort()


@langchain_cli.command()
@click.argument("document", type=click.Path(exists=True))
@click.option(
    "--provider",
    default="gemini",
    type=click.Choice(["gemini", "openai"]),
    help="LLM provider to use",
)
def analyze(document: str, provider: str):
    """Perform comprehensive document analysis."""
    try:
        analyzer = HybridDocumentAnalyzer(
            use_langchain=True,
            llm_provider=provider,
        )

        click.echo(f"Analyzing document: {document}\n")

        # Load document
        analyzer.load_document(document)

        # Perform analysis
        result = analyzer.analyze(provider=provider)

        click.echo(f"Summary:\n{result.summary}\n")
        click.echo(f"Sentiment: {result.sentiment_analysis.sentiment}")

        if result.key_points:
            click.echo(f"\nKey Points:")
            for i, point in enumerate(result.key_points, 1):
                click.echo(f"  {i}. {point.text}")

        if result.entities:
            click.echo(f"\nEntities:")
            for entity in result.entities:
                click.echo(f"  - {entity.text} ({entity.type})")

        if result.recommendations:
            click.echo(f"\nRecommendations:")
            for i, rec in enumerate(result.recommendations, 1):
                click.echo(f"  {i}. {rec.text}")

    except Exception as e:
        click.echo(f"✗ Error analyzing document: {e}", err=True)
        raise click.Abort()


@langchain_cli.command()
@click.argument("decision")
@click.option(
    "--context",
    default="",
    help="Context for the decision",
)
@click.option(
    "--provider",
    default="gemini",
    type=click.Choice(["gemini", "openai"]),
    help="LLM provider to use",
)
def explain(decision: str, context: str, provider: str):
    """Explain a decision with reasoning."""
    try:
        analyzer = HybridDocumentAnalyzer(
            use_langchain=True,
            llm_provider=provider,
        )

        click.echo(f"Explaining decision: {decision}\n")

        result = analyzer.explain_decision(decision, context=context)

        click.echo(f"Decision: {result.get('decision', decision)}")
        click.echo(f"Confidence: {result.get('confidence_score', 0):.2%}\n")

        if result.get("reasoning"):
            click.echo(f"Reasoning:")
            for i, reason in enumerate(result["reasoning"], 1):
                click.echo(f"  {i}. {reason}")

        if result.get("supporting_facts"):
            click.echo(f"\nSupporting Facts:")
            for fact in result["supporting_facts"]:
                click.echo(f"  - {fact}")

    except Exception as e:
        click.echo(f"✗ Error explaining decision: {e}", err=True)
        raise click.Abort()


@langchain_cli.command()
def config():
    """Show current configuration."""
    try:
        cfg = get_config()

        click.echo("LangChain Analyzer Configuration:")
        click.echo("================================")

        # Check for credentials (don't show actual values)
        try:
            gemini_key = cfg.get_gemini_key()
            gemini_status = "✓ Configured" if gemini_key and not gemini_key.startswith("YOUR_") else "✗ Not configured"
            click.echo(f"Gemini API: {gemini_status}")
        except:
            click.echo(f"Gemini API: ✗ Not configured")

        try:
            openai_key = cfg.get_openai_key()
            openai_status = "✓ Configured" if openai_key and not openai_key.startswith("YOUR_") else "✗ Not configured"
            click.echo(f"OpenAI API: {openai_status}")
        except:
            click.echo(f"OpenAI API: ✗ Not configured")

        click.echo("\nLangChain Settings:")
        click.echo(f"  Embedding Model: all-MiniLM-L6-v2")
        click.echo(f"  Chunk Size: 1000 characters")
        click.echo(f"  Chunk Overlap: 100 characters")
        click.echo(f"  Vector Store: Chroma")
        click.echo(f"  Retriever K: 5")

    except Exception as e:
        click.echo(f"✗ Error reading configuration: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    langchain_cli()
