"""
Root-level pytest configuration.

Patches tiktoken's BPE loader so that tests can run without network access.
The `encoding_for_model` call in `extract_all_messages.py` happens at module
import time; we must stub it out before any test module imports that file.
"""

import sys
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Mock tiktoken encoding so no network download is attempted.
# We patch both the top-level `tiktoken` entry-point and the internal
# `tiktoken.load.load_tiktoken_bpe` function that triggers the HTTP request.
# ---------------------------------------------------------------------------
_mock_encoding = MagicMock()
_mock_encoding.encode = lambda text: list(range(len(text.split())))  # simple word-count proxy

# Patch before any project module loads tiktoken
_tiktoken_patch = patch("tiktoken.encoding_for_model", return_value=_mock_encoding)
_tiktoken_patch.start()
