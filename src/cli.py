"""Command-line interface for document analyzer"""

import click
import json
import sys
from pathlib import Path
from .analyzer import DocumentAnalyzer
from .models import AnalysisRequest, DocumentType


@click.group()
def cli():
    """LLM-Based Document Analyzer"""
    pass


@cli.command()
@click.argument("document_path", type=click.Path(exists=True))
@click.option(
    "--doc-type",
    type=click.Choice(["pdf", "text", "markdown", "code", "email", "auto"]),
    default="auto",
    help="Document type hint",
)
@click.option(
    "--output",
    type=click.Path(),
    help="Output file path (JSON)",
)
@click.option(
    "--sentiment/--no-sentiment",
    default=True,
    help="Include sentiment analysis",
)
@click.option(
    "--recommendations/--no-recommendations",
    default=True,
    help="Include recommendations",
)
def analyze(document_path, doc_type, output, sentiment, recommendations):
    """Analyze a document"""
    try:
        # Read document
        with open(document_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Create analyzer
        analyzer = DocumentAnalyzer()

        # Prepare request
        doc_type_enum = None
        if doc_type != "auto":
            doc_type_enum = DocumentType(doc_type)

        request = AnalysisRequest(
            content=content,
            document_type=doc_type_enum,
            analyze_sentiment=sentiment,
            include_recommendations=recommendations,
        )

        # Run analysis
        click.echo("Analyzing document...", err=True)
        result = analyzer.analyze(request)

        # Format output
        output_data = result.model_dump(mode="json")

        if output:
            # Write to file
            with open(output, "w") as f:
                json.dump(output_data, f, indent=2)
            click.echo(f"Results saved to {output}")
        else:
            # Print to stdout
            click.echo(json.dumps(output_data, indent=2))

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--text", prompt=True, help="Text to analyze")
def quick_analyze(text):
    """Quick analysis of text"""
    try:
        analyzer = DocumentAnalyzer()
        request = AnalysisRequest(
            content=text,
            analyze_sentiment=True,
            include_recommendations=True,
        )
        result = analyzer.analyze(request)
        click.echo("\n=== Analysis Results ===")
        click.echo(f"Document Type: {result.document_type.value}")
        click.echo(f"Summary: {result.summary}")
        click.echo(
            f"Sentiment: {result.sentiment.overall_sentiment if result.sentiment else 'N/A'}"
        )
        click.echo(f"Processing Time: {result.processing_time:.2f}s")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def version():
    """Show version"""
    from . import __version__

    click.echo(f"LLM Document Analyzer v{__version__}")


if __name__ == "__main__":
    cli()
