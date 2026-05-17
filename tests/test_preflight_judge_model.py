# (c) JFrog Ltd. (2026)

"""Preflight regression tests for judge model resolution.

These tests pin the four invariants from the proposal's acceptance criteria:

1. ``--modes llm`` with no model in any layer fails preflight with the
   three-source error message.
2. ``--modes rules`` with no model in any layer succeeds (LLM-mode-only gate).
3. Model can come from any of the three layers (cli / env / yaml) on its
   own and still satisfies the gate.
4. The error message lists all three sources so the user does not need to
   hunt through docs to fix the failure.

These tests bypass argparse and call ``validate_scorers`` directly because
that is the boundary the eval / score CLI invokes during preflight; spinning
up the full CLI subprocess buys nothing for this contract.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from belt.commands.score import validate_scorers
from belt.errors import ConfigError


@contextmanager
def _isolated_llm_env():
    """Strip any BELT_LLM_* the developer's shell exported.

    Without this, a developer with ``BELT_LLM_MODEL`` in their shell would
    see these tests pass against the wrong layer; CI would still catch it,
    but the local feedback would mislead.
    """
    saved: dict[str, str] = {}
    for k in [k for k in os.environ if k.startswith("BELT_LLM_")]:
        saved[k] = os.environ.pop(k)
    try:
        yield
    finally:
        for k, v in saved.items():
            os.environ[k] = v


def _run_validate(
    *,
    cwd,
    modes: str,
    scorer_args: list[str] | None = None,
    yaml_body: str | None = None,
    env: dict[str, str] | None = None,
) -> list[str]:
    """Invoke ``validate_scorers`` with a controlled config layering.

    yaml_body, when given, is written to ``<cwd>/belt.yaml`` so the
    config-discovery walk picks it up; ``env`` is layered on top via patch.dict.

    Stubs ``BELT_OPENAI_API_KEY`` and ``BELT_ANTHROPIC_API_KEY`` so
    the backend availability check does not fail under CI (where the runner
    has no provider credentials). The contract being tested is the model
    gate; credential resolution and the judge-model preflight are separate
    concerns - the latter is exercised in
    ``tests/scorer/test_validate_scorers_preflight.py`` and in the
    per-backend probe tests under ``tests/scorer/llm/``. We pass
    ``probe_api=False`` here so a stub OpenAI key does not trigger a real
    HTTP call to api.openai.com.
    """
    if yaml_body is not None:
        (cwd / "belt.yaml").write_text(yaml_body)

    stubbed_creds = {
        "BELT_OPENAI_API_KEY": "sk-stub-for-tests",
        "BELT_ANTHROPIC_API_KEY": "sk-ant-stub-for-tests",
    }
    final_env = {**stubbed_creds, **(env or {})}

    with _isolated_llm_env():
        with patch.dict(os.environ, final_env, clear=False):
            old_cwd = os.getcwd()
            os.chdir(cwd)
            try:
                return validate_scorers(modes, scorer_args, None, probe_api=False)
            finally:
                os.chdir(old_cwd)


class TestModesRulesIgnoresMissingModel:
    """Acceptance criterion: ``--modes rules`` succeeds without a model."""

    def test_rules_only_no_model_anywhere(self, tmp_path):
        # No yaml, no env, no CLI scorer-args - the rule scorer should still build.
        descriptions = _run_validate(cwd=tmp_path, modes="rules")
        assert any("rules" in d for d in descriptions)


class TestModesLlmRequiresModel:
    """Acceptance criterion: ``--modes llm`` fails preflight without a model."""

    def test_llm_no_model_anywhere_raises_config_error(self, tmp_path):
        with pytest.raises(ConfigError) as exc_info:
            _run_validate(cwd=tmp_path, modes="llm")
        msg = str(exc_info.value)
        # All three sources surface in the error so the user can self-serve the fix.
        assert "--scorer-arg model=" in msg
        assert "BELT_LLM_MODEL" in msg
        assert "belt.yaml" in msg

    def test_rules_and_llm_no_model_raises(self, tmp_path):
        # Combined modes still trips the gate because llm is in the set.
        with pytest.raises(ConfigError):
            _run_validate(cwd=tmp_path, modes="rules,llm")

    def test_yaml_only_satisfies_gate(self, tmp_path):
        descriptions = _run_validate(
            cwd=tmp_path,
            modes="llm",
            yaml_body="llm:\n  model: openai/gpt-5.4-mini\n",
        )
        assert any("openai/gpt-5.4-mini" in d for d in descriptions)

    def test_env_only_satisfies_gate(self, tmp_path):
        descriptions = _run_validate(
            cwd=tmp_path,
            modes="llm",
            env={"BELT_LLM_MODEL": "openai/gpt-5.4-mini"},
        )
        assert any("openai/gpt-5.4-mini" in d for d in descriptions)

    def test_cli_only_satisfies_gate(self, tmp_path):
        descriptions = _run_validate(
            cwd=tmp_path,
            modes="llm",
            scorer_args=["model=openai/gpt-5.4-mini"],
        )
        assert any("openai/gpt-5.4-mini" in d for d in descriptions)
