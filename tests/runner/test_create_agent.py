# (c) JFrog Ltd. (2026)

"""Tests for create_agent() and env-var resolution."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

from belt.agent.base import AgentArgError, AgentOption, BaseAgentAdapter
from belt.commands.run import _resolve_env_var_defaults, create_agent
from belt.entities import TurnOutput
from belt.runner.entities import AgentConfig


class FakeAgentAdapter(BaseAgentAdapter):
    """Agent that accepts a 'model' option with env_var fallback."""

    def __init__(self, *, model: str | None = None, **_kwargs: Any):
        self.model = model

    @classmethod
    def cli_options(cls) -> list[AgentOption]:
        return [AgentOption(name="model", help="Model override", env_var="FAKE_MODEL")]

    def setup(self, config: AgentConfig) -> None:
        pass

    def execute(self, message: str, flags: list[str]) -> str:
        return ""

    def fetch_results(self, raw_output: str) -> TurnOutput:
        return TurnOutput(raw_cli=raw_output, reply_text="", has_reply=False, has_error=False)

    def teardown(self) -> None:
        pass


class NoOptionsAgentAdapter(BaseAgentAdapter):
    """Agent with no cli_options declared."""

    def __init__(self, **_kwargs: Any):
        pass

    def setup(self, config: AgentConfig) -> None:
        pass

    def execute(self, message: str, flags: list[str]) -> str:
        return ""

    def fetch_results(self, raw_output: str) -> TurnOutput:
        return TurnOutput(raw_cli=raw_output, reply_text="", has_reply=False, has_error=False)

    def teardown(self) -> None:
        pass


class TestResolveEnvVarDefaults:
    def test_env_var_injected_when_not_in_args(self) -> None:
        with patch.dict(os.environ, {"FAKE_MODEL": "gpt-4.1-mini"}):
            result = _resolve_env_var_defaults(FakeAgentAdapter, {})
        assert result["model"] == "gpt-4.1-mini"

    def test_explicit_arg_takes_precedence_over_env(self) -> None:
        with patch.dict(os.environ, {"FAKE_MODEL": "from-env"}):
            result = _resolve_env_var_defaults(FakeAgentAdapter, {"model": "from-flag"})
        assert result["model"] == "from-flag"

    def test_unset_env_var_ignored(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = _resolve_env_var_defaults(FakeAgentAdapter, {})
        assert "model" not in result

    def test_empty_env_var_ignored(self) -> None:
        with patch.dict(os.environ, {"FAKE_MODEL": ""}):
            result = _resolve_env_var_defaults(FakeAgentAdapter, {})
        assert "model" not in result

    def test_no_cli_options_is_noop(self) -> None:
        with patch.dict(os.environ, {"FAKE_MODEL": "should-not-matter"}):
            result = _resolve_env_var_defaults(NoOptionsAgentAdapter, {"repo_root": "/tmp"})
        assert result == {"repo_root": "/tmp"}

    def test_original_dict_not_mutated(self) -> None:
        original = {"repo_root": "/tmp"}
        with patch.dict(os.environ, {"FAKE_MODEL": "injected"}):
            result = _resolve_env_var_defaults(FakeAgentAdapter, original)
        assert "model" not in original
        assert result["model"] == "injected"


class TestCreateAgentWithEnvVar:
    def test_agent_receives_env_var_model(self) -> None:
        with patch.dict(os.environ, {"FAKE_MODEL": "gpt-4.1-mini"}):
            agent = create_agent(FakeAgentAdapter, {"repo_root": "/tmp"})
        assert isinstance(agent, FakeAgentAdapter)
        assert agent.model == "gpt-4.1-mini"

    def test_flag_overrides_env_var(self) -> None:
        with patch.dict(os.environ, {"FAKE_MODEL": "from-env"}):
            agent = create_agent(FakeAgentAdapter, {"model": "from-flag", "repo_root": "/tmp"})
        assert agent.model == "from-flag"

    def test_no_env_no_flag_uses_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            agent = create_agent(FakeAgentAdapter, {"repo_root": "/tmp"})
        assert agent.model is None

    def test_unknown_arg_still_rejected(self) -> None:
        with pytest.raises(AgentArgError, match="does not accept"):
            create_agent(FakeAgentAdapter, {"bad_option": "val", "repo_root": "/tmp"})


class ParameterlessAgentAdapter(BaseAgentAdapter):
    """Agent that mirrors the built-in parameterless agents: no ``__init__``
    kwargs and no declared ``cli_options()``. Used to exercise the
    "rejects every -X arg with a friendly message" contract in
    ``create_agent``."""

    def __init__(self) -> None:
        pass

    def setup(self, config: AgentConfig) -> None:
        pass

    def execute(self, message: str, flags: list[str]) -> str:
        return ""

    def fetch_results(self, raw_output: str) -> TurnOutput:
        return TurnOutput(raw_cli=raw_output, reply_text="", has_reply=False, has_error=False)

    def teardown(self) -> None:
        pass


class TestParameterlessAdapterRejectsXArgs:
    """`-X foo=bar` against a parameterless agent must produce a friendly,
    framework-shaped error, never a rewrapped Python TypeError."""

    def test_no_x_args_constructs_cleanly(self) -> None:
        agent = create_agent(ParameterlessAgentAdapter, {"repo_root": "/tmp"})
        assert isinstance(agent, ParameterlessAgentAdapter)

    def test_x_arg_rejected_with_friendly_message(self) -> None:
        with pytest.raises(AgentArgError) as exc:
            create_agent(ParameterlessAgentAdapter, {"model": "claude-sonnet-4", "repo_root": "/tmp"})
        msg = str(exc.value)
        assert "does not accept any -X options" in msg
        assert "model" in msg
        assert "parameterless by design" in msg
        assert "scenario" in msg
        assert "got an unexpected keyword argument" not in msg, (
            "Parameterless agent is leaking Python TypeError detail; "
            "create_agent() should reject unknown -X args before they reach __init__."
        )

    def test_multiple_x_args_listed_in_error(self) -> None:
        with pytest.raises(AgentArgError) as exc:
            create_agent(
                ParameterlessAgentAdapter,
                {"model": "x", "temperature": "0.7", "repo_root": "/tmp"},
            )
        msg = str(exc.value)
        assert "model" in msg and "temperature" in msg


class TestErrorShapeAcrossRegistry:
    """Every registered built-in agent must reject unknown -X args with the
    framework's standard ``AgentArgError`` format, never a rewrapped
    Python ``TypeError`` message that leaks ``__init__`` internals.

    Auto-discovery over ``_AGENT_REGISTRY`` guarantees that any built-in
    added later is covered automatically the moment it is registered, so
    the contract cannot regress silently when the registry grows.
    """

    @pytest.mark.parametrize(
        "agent_name",
        sorted(__import__("belt.agent.registry", fromlist=["_AGENT_REGISTRY"])._AGENT_REGISTRY),
    )
    def test_unknown_x_arg_produces_framework_shaped_error(self, agent_name: str) -> None:
        from belt.agent.registry import get_agent_class

        cls = get_agent_class(agent_name)
        with pytest.raises(AgentArgError) as exc:
            create_agent(cls, {"definitely_not_a_real_option": "x", "repo_root": "/tmp"})
        msg = str(exc.value)
        assert "got an unexpected keyword argument" not in msg, (
            f"Agent '{agent_name}' leaks Python TypeError detail into AgentArgError. "
            f"create_agent() should reject unknown -X args before they reach __init__. "
            f"Got: {msg}"
        )
        assert "does not accept" in msg, (
            f"Agent '{agent_name}' did not produce the framework's standard "
            f"'does not accept' error format. Got: {msg}"
        )
