# (c) JFrog Ltd. (2026)

"""Pin the canonical-example model centralization.

The recommended LLM model for CLI help, error messages, and the doctor
diagnostic must come from a single constant - ``belt.constants.EXAMPLE_LLM_MODEL``.
Without this test, contributors re-add hard-coded model names to those
surfaces and the constant silently drifts out of sync.

Standalone markdown examples are explicitly out of scope here: docs are
illustrative, the constant is normative. See the comment on
``EXAMPLE_LLM_MODEL`` in ``constants.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from belt.constants import EXAMPLE_LLM_MODEL

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "belt"


# Code paths that must reference EXAMPLE_LLM_MODEL only via the constant -
# no hard-coded ``openai/gpt-5.4-mini`` (or any other ``openai/<name>``)
# string for the *recommended* model. Module docstrings / illustrative
# format examples (``parse_model_spec`` docstring, header comments)
# are exempted because they explain the *prefix* shape, not the choice.
_NORMATIVE_FILES = [
    "config.py",
    "commands/doctor.py",
    "commands/eval.py",
    "commands/quickstart.py",
    "commands/score.py",
]


@pytest.fixture(scope="module")
def constant_value() -> str:
    return EXAMPLE_LLM_MODEL


def test_constant_is_provider_prefixed(constant_value: str):
    """Sanity: the constant must include a provider prefix to route correctly."""
    assert "/" in constant_value, f"EXAMPLE_LLM_MODEL must be prefixed (got {constant_value!r})"
    prefix = constant_value.split("/", 1)[0]
    assert prefix in {"openai", "azure", "anthropic", "ollama"}


@pytest.mark.parametrize("relative_path", _NORMATIVE_FILES)
def test_normative_surface_has_no_hardcoded_recommended_model(relative_path: str, constant_value: str):
    """Files that build user-facing error / help text must reference the constant.

    Concretely: any literal ``openai/<name>`` substring in any of these files
    is a regression - it should be either ``EXAMPLE_LLM_MODEL`` or an f-string
    interpolation of it. The placeholder form ``openai/...`` (literal three
    dots) is exempted because module docstrings use it to explain the prefix
    shape rather than to recommend a specific model.

    Why drop the quote requirement: argparse help strings embed the literal
    inside a Python string without inner quotes (``"--agent-arg
    model=openai/gpt-X"``), which the previous quote-anchored regex missed.
    Substring matching catches both quoted and unquoted hard-codes.
    """
    text = (SRC_ROOT / relative_path).read_text()

    # ``openai/(?!\.\.\.)<name>`` matches a concrete model name but not the
    # placeholder ``openai/...``. ``[a-z0-9._-]+`` is the allowed model-name
    # alphabet; the negative lookahead skips three-dot placeholders that
    # docstrings use to describe the prefix shape.
    pattern = re.compile(r"openai/(?!\.\.\.)[a-z0-9._-]+", re.IGNORECASE)
    matches = pattern.findall(text)
    # The constant value itself contains the literal once (in constants.py),
    # but constants.py is intentionally not in _NORMATIVE_FILES.
    assert not matches, (
        f"{relative_path} contains hard-coded model name(s) {matches}; "
        f"reference `EXAMPLE_LLM_MODEL` from `belt.constants` instead "
        f"(use an f-string for argparse help and error messages)."
    )


def test_constant_appears_in_no_model_hint(constant_value: str):
    """The three-source error message must f-string from the constant."""
    from belt.config import _NO_MODEL_HINT

    occurrences = _NO_MODEL_HINT.count(constant_value)
    # CLI flag, env var, yaml - three distinct lines all reference it.
    assert occurrences == 3, (
        f"_NO_MODEL_HINT should reference EXAMPLE_LLM_MODEL three times "
        f"(once per source line); got {occurrences}.\nMessage:\n{_NO_MODEL_HINT}"
    )


def test_constant_appears_in_doctor_judge_model_suggestion(constant_value: str, monkeypatch, tmp_path):
    """doctor's `(not set)` suggestion uses the constant in all three lines."""
    monkeypatch.chdir(tmp_path)
    import os as _os

    for k in [k for k in list(_os.environ) if k.startswith("BELT_LLM_")]:
        monkeypatch.delenv(k, raising=False)

    from belt.commands.doctor import _check_judge_model

    result = _check_judge_model()
    assert not result.ok
    assert result.suggestion.count(constant_value) == 3


def test_constant_appears_in_scorer_cli_help(constant_value: str):
    """The ``-S/--scorer-arg`` argparse help references the constant."""
    from belt.commands.score import LLM_SCORER_OPTIONS

    model_option = next(opt for opt in LLM_SCORER_OPTIONS if opt.name == "model")
    assert constant_value in model_option.help, (
        f"LLM_SCORER_OPTIONS 'model' help should mention EXAMPLE_LLM_MODEL; " f"got: {model_option.help!r}"
    )
