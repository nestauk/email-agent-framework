"""Tests for authentication module."""

import pytest

from email_agent.agent_api.auth import get_agent_api_key


def test_get_agent_api_key_raises_when_missing(monkeypatch):
    """Verify that missing AGENT_API_KEY raises RuntimeError."""
    monkeypatch.delenv("AGENT_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="AGENT_API_KEY"):
        get_agent_api_key()


def test_get_agent_api_key_returns_value_when_set(monkeypatch):
    """Verify that set AGENT_API_KEY is returned."""
    monkeypatch.setenv("AGENT_API_KEY", "test-key-123")
    assert get_agent_api_key() == "test-key-123"
