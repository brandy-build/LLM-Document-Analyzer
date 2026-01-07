"""Enhanced CLI with PDF support, Q&A, and decision explanation"""

import click
import json
import sys
from pathlib import Path
from src.enhanced_analyzer import EnhancedDocumentAnalyzer
from src.models import AnalysisRequest


@click.group()
def enhanced_cli():
    """Enhanced LLM Document Analyzer with PDF support and Q&A"""
    pass


@enhanced_cli.command()
@click.argument("document_path", type=click.Path(exists=True))
@click.option("--provider", type=click.Choice(["gemini", "openai"]), default="gemini", help="AI provider")
@click.option("--use-embeddings", is_flag=True, help="Use embeddings for analysis")
@click.option("--output", type=click.Path(), help="Output file path (JSON)")
def analyze(document_path, provider, use_embeddings, output):
    """Analyze a document (supports PDF, TXT, MD)"""
    try:
        analyzer = EnhancedDocumentAnalyzer(default_provider=provider)

        click.echo(f"Analyzing {Path(document_path).name}...", err=True)

        request = AnalysisRequest(
            file_path=document_path,
            use_embeddings=use_embeddings,
            ai_provider=provider,
        )

        result = analyzer.analyze(request)
        output_data = result.model_dump(mode="json")

        if output:
            with open(output, "w") as f:
                json.dump(output_data, f, indent=2)
            click.echo(f"Results saved to {output}")
        else:
            click.echo(json.dumps(output_data, indent=2))

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@enhanced_cli.command()
@click.argument("document_path", type=click.Path(exists=True))
@click.option("--provider", type=click.Choice(["gemini", "openai"]), default="gemini", help="AI provider")
@click.option("--use-embeddings", is_flag=True, help="Use embeddings for semantic search")
@click.option("--output", type=click.Path(), help="Output file path (JSON)")
def ask(document_path, provider, use_embeddings, output):
    """Ask questions about a document"""
    try:
        analyzer = EnhancedDocumentAnalyzer(default_provider=provider)

        # Load document
        content = analyzer.load_document(document_path)

        # Build embeddings if requested
        if use_embeddings:
            click.echo("Building embedding index...", err=True)
            analyzer.build_embedding_index(content)

        # Interactive Q&A
        click.echo("\n=== Q&A Mode (type 'exit' to quit) ===\n")
        qa_results = []

        while True:
            question = click.prompt("Question")
            if question.lower() == "exit":
                break

            try:
                result = analyzer.answer_question(
                    question, use_embeddings=use_embeddings, provider=provider
                )
                click.echo(f"\nAnswer: {result.answer}")

                if result.citations:
                    click.echo("\nCitations:")
                    for citation in result.citations:
                        click.echo(f"  - {citation}")

                click.echo(f"Confidence: {result.confidence:.2%}\n")
                qa_results.append(result.model_dump(mode="json"))

            except Exception as e:
                click.echo(f"Error: {e}", err=True)

        if output and qa_results:
            with open(output, "w") as f:
                json.dump(qa_results, f, indent=2)
            click.echo(f"Q&A results saved to {output}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@enhanced_cli.command()
@click.argument("document_path", type=click.Path(exists=True))
@click.option("--decision", prompt="Decision to explain", help="Decision to explain")
@click.option("--provider", type=click.Choice(["gemini", "openai"]), default="gemini", help="AI provider")
@click.option("--output", type=click.Path(), help="Output file path (JSON)")
def explain(document_path, decision, provider, output):
    """Explain why a decision was made"""
    try:
        analyzer = EnhancedDocumentAnalyzer(default_provider=provider)

        click.echo(f"Analyzing document for decision explanation...", err=True)
        content = analyzer.load_document(document_path)

        explanation = analyzer.explain_decision(decision, content[:5000], provider=provider)

        click.echo("\n=== Decision Explanation ===")
        click.echo(f"Decision: {explanation['decision']}")
        click.echo(f"\nExplanation:\n{explanation['explanation']}")

        if output:
            with open(output, "w") as f:
                json.dump(explanation, f, indent=2)
            click.echo(f"\nResults saved to {output}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@enhanced_cli.command()
@click.argument("document_path", type=click.Path(exists=True))
@click.option("--provider", type=click.Choice(["gemini", "openai"]), default="gemini", help="AI provider")
@click.option("--output", type=click.Path(), help="Output file path (JSON)")
def extract_pdf(document_path, provider, output):
    """Extract and summarize PDF content"""
    try:
        from src.document_processor import PDFProcessor

        click.echo(f"Extracting PDF content...", err=True)
        content = PDFProcessor.extract_text_from_pdf(document_path)

        click.echo(f"\nExtracted {len(content)} characters")
        click.echo(f"\nPreview:\n{content[:500]}...\n")

        if output:
            with open(output, "w") as f:
                f.write(content)
            click.echo(f"Content saved to {output}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@enhanced_cli.command()
def version():
    """Show version"""
    from src import __version__

    click.echo(f"Enhanced LLM Document Analyzer v{__version__}")


if __name__ == "__main__":
    enhanced_cli()
