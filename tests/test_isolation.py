"""The suite's own safety net.

Every other test depends on the network being unreachable and no API key being set.
If those fixtures silently stopped working, tests would start making real calls and
still pass, so the guards are asserted directly.
"""

import os
import socket

import pytest


class TestNetworkGuard:
    def test_outbound_connections_are_blocked(self):
        with pytest.raises(RuntimeError, match="must not reach the network"):
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("example.com", 80))

    def test_the_error_explains_what_to_fix(self):
        with pytest.raises(RuntimeError, match="missing its fake"):
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))


class TestCredentialGuard:
    def test_no_api_key_is_visible(self):
        assert os.getenv("GOOGLE_API_KEY") is None
        assert os.getenv("GEMINI_API_KEY") is None

    def test_the_real_embedding_client_refuses_to_build(self):
        # Proof that nothing reaches the embedding API by accident: without a key it
        # raises rather than falling back to some default credential.
        from arxiv_reviewer.rag import get_embeddings

        get_embeddings.cache_clear()
        with pytest.raises(RuntimeError, match="not set"):
            get_embeddings()
        get_embeddings.cache_clear()

    def test_the_real_chat_client_refuses_to_build(self):
        from arxiv_reviewer.gemini_client import gemini_llm

        with pytest.raises(RuntimeError, match="not set"):
            gemini_llm()
