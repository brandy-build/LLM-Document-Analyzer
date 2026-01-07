"""Tests configuration"""

import pytest
import os
from unittest.mock import patch


@pytest.fixture(autouse=True)
def mock_api_key():
    """Auto-mock API key for all tests"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        yield
