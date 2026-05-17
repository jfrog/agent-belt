# (c) JFrog Ltd. (2026)

"""Tests for agent registry."""

from __future__ import annotations

import pytest

from belt.agent.registry import available_agents, get_agent_class


def test_available_agents_returns_known_names() -> None:
    names = available_agents()
    assert "claude-code" in names
    assert "cursor" in names
    assert "codex" in names


def test_unknown_agent_raises() -> None:
    from belt.errors import ConfigError

    with pytest.raises(ConfigError, match="Unknown agent 'nonexistent'"):
        get_agent_class("nonexistent")


def test_invalid_dotted_path_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dotted-path escape hatch surfaces an import error when the user opts in."""
    from belt.errors import ConfigError

    monkeypatch.setenv("BELT_ALLOW_ARBITRARY_AGENT", "1")
    with pytest.raises(ConfigError, match="Failed to load agent"):
        get_agent_class("totally.bogus.Module")


def test_non_agent_class_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from belt.errors import ConfigError

    monkeypatch.setenv("BELT_ALLOW_ARBITRARY_AGENT", "1")
    with pytest.raises(ConfigError, match="not a BaseAgentAdapter subclass"):
        get_agent_class("belt.entities.ToolCall")


def test_dotted_path_blocked_without_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dotted-import escape hatch is off by default."""
    from belt.errors import ConfigError

    monkeypatch.delenv("BELT_ALLOW_ARBITRARY_AGENT", raising=False)
    with pytest.raises(ConfigError, match="Unknown agent 'totally.bogus.Module'"):
        get_agent_class("totally.bogus.Module")
