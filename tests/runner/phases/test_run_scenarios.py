# (c) JFrog Ltd. (2026)

"""Tests for ``runner.phases.run_scenarios`` - in particular the
implicit-default-model warning that protects users from opaque
agent-CLI failures when no explicit model is set."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from belt.agent.base import AgentOption, BaseAgentAdapter
from belt.entities import TurnOutput
from belt.runner.context import MatchedGroup, RunContext
from belt.runner.entities import AgentConfig
from belt.runner.phases.run_scenarios import emit_implicit_default_model_warnings
from belt.scenario import GroupConfig


class _ModelAgentAdapter(BaseAgentAdapter):
    """Agent that exposes a 'model' option with an env-var fallback."""

    def __init__(self, *, model: str | None = None, **_kwargs: Any):
        self.model = model

    @classmethod
    def cli_options(cls) -> list[AgentOption]:
        return [AgentOption(name="model", help="Model", env_var="FAKE_MODEL")]

    def setup(self, config: AgentConfig) -> None: ...
    def execute(self, message: str, flags: list[str]) -> str:
        return ""

    def fetch_results(self, raw_output: str) -> TurnOutput:
        return TurnOutput(raw_cli=raw_output)

    def teardown(self) -> None: ...


class _NoModelAgentAdapter(BaseAgentAdapter):
    """Agent that does NOT expose a 'model' option."""

    @classmethod
    def cli_options(cls) -> list[AgentOption]:
        return []

    def setup(self, config: AgentConfig) -> None: ...
    def execute(self, message: str, flags: list[str]) -> str:
        return ""

    def fetch_results(self, raw_output: str) -> TurnOutput:
        return TurnOutput(raw_cli=raw_output)

    def teardown(self) -> None: ...


def _make_ctx(
    agent_name: str,
    agent_args: dict[str, str],
    *,
    n_groups: int = 1,
) -> RunContext:
    """Minimal RunContext with one or more matched groups using ``agent_name``."""
    matched: list[MatchedGroup] = []
    for i in range(n_groups):
        gc = GroupConfig(agent=agent_name)
        matched.append(MatchedGroup(group_dir=Path(f"/tmp/group_{i}"), config=gc, scenarios=[], name=f"g{i}"))
    progress = MagicMock()
    return RunContext(
        args=MagicMock(),
        scenarios_root=Path("/tmp"),
        matched_groups=matched,
        agent_args=agent_args,
        outcomes_root=Path("/tmp/out"),
        run_dir=Path("/tmp/out/run"),
        workspace=Path("/tmp"),
        progress=progress,
    )


@pytest.fixture
def _registry_with_model_agent():
    with patch("belt.runner.phases.run_scenarios.get_agent_class") as gac:
        gac.return_value = _ModelAgentAdapter
        yield gac


@pytest.fixture
def _registry_with_no_model_agent():
    with patch("belt.runner.phases.run_scenarios.get_agent_class") as gac:
        gac.return_value = _NoModelAgentAdapter
        yield gac


@pytest.fixture(autouse=True)
def _clear_fake_model_env():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("FAKE_MODEL", None)
        yield


class TestEmitImplicitDefaultModelWarnings:
    def test_warns_when_no_explicit_model(self, _registry_with_model_agent, caplog) -> None:
        ctx = _make_ctx("fake", agent_args={})
        warned = emit_implicit_default_model_warnings(ctx)
        assert warned == ["fake"]

    def test_silent_when_x_flag_sets_model(self, _registry_with_model_agent) -> None:
        ctx = _make_ctx("fake", agent_args={"model": "openai/gpt-5.4-mini"})
        assert emit_implicit_default_model_warnings(ctx) == []

    def test_silent_when_env_var_sets_model(self, _registry_with_model_agent) -> None:
        with patch.dict(os.environ, {"FAKE_MODEL": "openai/gpt-5.4-mini"}):
            ctx = _make_ctx("fake", agent_args={})
            assert emit_implicit_default_model_warnings(ctx) == []

    def test_silent_when_agent_has_no_model_option(self, _registry_with_no_model_agent) -> None:
        ctx = _make_ctx("nomodel", agent_args={})
        assert emit_implicit_default_model_warnings(ctx) == []

    def test_warns_once_per_agent_with_repeat_groups(self, _registry_with_model_agent) -> None:
        ctx = _make_ctx("fake", agent_args={}, n_groups=3)
        warned = emit_implicit_default_model_warnings(ctx)
        assert warned == ["fake"]  # 3 groups, but 1 warning

    def test_unknown_agent_does_not_raise(self) -> None:
        with patch(
            "belt.runner.phases.run_scenarios.get_agent_class",
            side_effect=KeyError("unknown"),
        ):
            ctx = _make_ctx("missing", agent_args={})
            assert emit_implicit_default_model_warnings(ctx) == []


# Tests for the external-working-dir guardrail moved to
# ``tests/runner/phases/test_setup_groups.py`` when the check was
# promoted from a per-scenario raise (in ``run_scenarios``) to a
# per-group gate (in ``setup_groups``). The function name there is
# ``_external_working_dir_message``.
