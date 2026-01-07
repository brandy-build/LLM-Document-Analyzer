"""
Enhanced Examples - LLM Document Analyzer v2.0
Shows new features: PDF support, embeddings, Q&A with citations, Gemini AI
"""

import sys
from src.enhanced_analyzer import EnhancedDocumentAnalyzer
from src.models import AnalysisRequest
from src.document_processor import PDFProcessor


def example_pdf_analysis():
    """Example: Analyze a PDF document"""
    print("\n" + "="*60)
    print("Example 1: PDF Document Analysis")
    print("="*60)

    # Note: In real usage, provide an actual PDF file path
    pdf_content = """
    COMPANY POLICY DOCUMENT v2.0
    
    1. REMOTE WORK POLICY
    Employees may work remotely up to 3 days per week with manager approval.
    
    2. LEAVE POLICY  
    - Annual leave: 20 days per year
    - Sick leave: 10 days per year
    - Maternity leave: 6 months
    
    3. BENEFITS
    - Health insurance coverage
    - Retirement plan matching
    - Professional development budget
    
    4. CODE OF CONDUCT
    All employees must adhere to ethical guidelines and company values.
    """

    print("\nPDF Content Preview:")
    print(pdf_content[:200] + "...")

    # In real scenario:
    # analyzer = EnhancedDocumentAnalyzer(gemini_api_key="your_key")
    # content = analyzer.load_document("policy.pdf")


def example_embeddings_search():
    """Example: Build embeddings and semantic search"""
    print("\n" + "="*60)
    print("Example 2: Semantic Search with Embeddings")
    print("="*60)

    print("\nHow embeddings work:")
    print("1. Document is chunked into segments")
    print("2. Each chunk is converted to a vector")
    print("3. Vectors are stored in embedding index")
    print("4. User query is embedded")
    print("5. Similar chunks are retrieved")
    print("6. Retrieved chunks provide context for Q&A")

    print("\nExample chunks:")
    chunks = [
        "Remote work is allowed 3 days per week",
        "Annual leave is 20 days per year",
        "Health insurance is provided",
        "Professional development budget available",
    ]

    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk}")

    print("\nWith embeddings, this query:")
    print('  "How much leave do I get?"')
    print("Would retrieve the most relevant chunks")
    print("and provide accurate answers with citations.")


def example_qa_with_citations():
    """Example: Q&A with source citations"""
    print("\n" + "="*60)
    print("Example 3: Q&A with Citations")
    print("="*60)

    example_qa = {
        "question": "What is the remote work policy?",
        "answer": "Employees may work remotely up to 3 days per week with manager approval.",
        "citations": [
            {
                "text": "Employees may work remotely up to 3 days per week",
                "chunk_id": 5,
                "page": 2,
                "confidence": 0.95,
            }
        ],
        "confidence": 0.92,
    }

    print(f"\nQuestion: {example_qa['question']}")
    print(f"Answer: {example_qa['answer']}")
    print(f"Confidence: {example_qa['confidence']:.1%}")
    print("\nCitations:")
    for citation in example_qa["citations"]:
        print(
            f'  Source: "{citation["text"]}" '
            f"(Chunk {citation['chunk_id']}, Page {citation['page']}, "
            f"Confidence: {citation['confidence']:.0%})"
        )


def example_decision_explanation():
    """Example: Explain why a decision was made"""
    print("\n" + "="*60)
    print("Example 4: Decision Explanation")
    print("="*60)

    print("\nDecision to Explain:")
    print('"The company implemented a hybrid remote work policy"')

    print("\nExpected Explanation:")
    print("1. Context: Modern companies need flexibility")
    print("2. Reason: Improve employee satisfaction")
    print("3. Evidence: Industry trends show hybrid work increases productivity")
    print("4. Factors:")
    print("   - Employee preferences")
    print("   - Market competitiveness")
    print("   - Cost savings")
    print("   - Work-life balance")

    print("\nWith Gemini, it explains:")
    print("- What decision was made")
    print("- Why it was made")
    print("- Supporting factors and evidence")


def example_gemini_vs_openai():
    """Example: Compare Gemini and OpenAI providers"""
    print("\n" + "="*60)
    print("Example 5: Gemini vs OpenAI Provider Comparison")
    print("="*60)

    comparison = {
        "Gemini": {
            "pros": [
                "Free tier available",
                "Excellent reasoning",
                "Fast processing",
                "Good for explanations",
            ],
            "cost": "Free tier / Paid",
            "best_for": "PDF analysis, decision explanations",
        },
        "OpenAI": {
            "pros": [
                "Mature API",
                "GPT-4 available",
                "Extensive customization",
                "Good for coding",
            ],
            "cost": "Pay-as-you-go",
            "best_for": "Complex analysis, specialized tasks",
        },
    }

    for provider, details in comparison.items():
        print(f"\n{provider}:")
        print(f"  Best for: {details['best_for']}")
        print(f"  Cost: {details['cost']}")
        print("  Strengths:")
        for pro in details["pros"]:
            print(f"    - {pro}")


def example_real_workflow():
    """Example: Real workflow for policy analysis"""
    print("\n" + "="*60)
    print("Example 6: Complete Workflow")
    print("="*60)

    print("""
Workflow: Analyze Company Policy Document

1. LOAD DOCUMENT
   analyzer = EnhancedDocumentAnalyzer(gemini_api_key="key")
   content = analyzer.load_document("company_policy.pdf")

2. BUILD EMBEDDINGS
   analyzer.build_embedding_index(content)
   -> Creates 50+ chunks with embeddings

3. ASK QUESTIONS
   q1 = analyzer.answer_question("What is remote work policy?")
   q2 = analyzer.answer_question("How many vacation days?")

4. GET CITATIONS
   for citation in q1.citations:
       print(citation)  # Shows source with page numbers

5. EXPLAIN DECISIONS
   explanation = analyzer.explain_decision(
       decision="Policy was updated in 2023",
       context=content
   )

6. EXPORT RESULTS
   results = {
       "questions": [q1, q2],
       "explanations": [explanation],
       "metadata": result.metadata
   }
   save_json("analysis_results.json", results)
    """)


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("ENHANCED LLM DOCUMENT ANALYZER - v2.0 Examples")
    print("="*60)

    example_pdf_analysis()
    example_embeddings_search()
    example_qa_with_citations()
    example_decision_explanation()
    example_gemini_vs_openai()
    example_real_workflow()

    print("\n" + "="*60)
    print("To use these features in real code:")
    print("="*60)
    print("""
1. Set your Gemini API key:
   export GOOGLE_API_KEY="your_key"
   
2. Create analyzer:
   from src.enhanced_analyzer import EnhancedDocumentAnalyzer
   analyzer = EnhancedDocumentAnalyzer()

3. Analyze PDF:
   content = analyzer.load_document("policy.pdf")
   analyzer.build_embedding_index(content)

4. Ask questions:
   result = analyzer.answer_question("Your question here")

5. Get citations:
   for citation in result.citations:
       print(citation.text, "page", citation.page)

For CLI usage:
   python -m src.enhanced_cli analyze policy.pdf
   python -m src.enhanced_cli ask policy.pdf
   python -m src.enhanced_cli explain policy.pdf
    """)

    print("="*60)
    print("See ENHANCED_FEATURES.md for detailed documentation")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
