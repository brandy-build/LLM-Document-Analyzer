#!/usr/bin/env python
"""Quick test to verify LangChain integration setup."""

import sys

print("=" * 60)
print("LangChain Integration Setup Verification")
print("=" * 60)

# Test 1: Core imports
print("\n[1] Testing core analyzer imports...")
try:
    from src import EnhancedDocumentAnalyzer, DocumentAnalyzer
    print("    ✓ Core analyzers imported successfully")
except Exception as e:
    print(f"    ✗ Error: {e}")
    sys.exit(1)

# Test 2: Security config
print("\n[2] Testing secure config...")
try:
    from src import get_config
    config = get_config()
    print("    ✓ SecureConfig initialized")
except Exception as e:
    print(f"    ✗ Error: {e}")
    sys.exit(1)

# Test 3: LangChain components (optional - may have dependency issues)
print("\n[3] Testing LangChain components (optional)...")
try:
    from src import LangChainDocumentAnalyzer, HybridDocumentAnalyzer
    print("    ✓ LangChain components imported successfully")
    print("    ✓ LangChain integration fully installed")
except ImportError as e:
    print(f"    ⚠ LangChain components not available (expected if not fully installed)")
    print(f"    Install with: pip install langchain langchain-community langchain-google-genai langchain-openai")
except Exception as e:
    print(f"    ✗ Error: {e}")

# Test 4: Version check
print("\n[4] Checking package versions...")
try:
    import src
    print(f"    ✓ Package version: {src.__version__}")
except Exception as e:
    print(f"    ✗ Error: {e}")

print("\n" + "=" * 60)
print("Setup verification complete!")
print("=" * 60)

print("\nNext steps:")
print("  1. Original API: from src import EnhancedDocumentAnalyzer")
print("  2. LangChain API: from src import LangChainDocumentAnalyzer")
print("  3. Hybrid API (recommended): from src import HybridDocumentAnalyzer")
print("\nDocumentation: docs/LANGCHAIN_GUIDE.md")
