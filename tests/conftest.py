# (c) JFrog Ltd. (2026)

"""Add src/ to sys.path so tests can import framework modules."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_BELT_ENV_VARS = [
    "BELT_LLM_MODEL",
    "BELT_LLM_PROVIDER",
]


@pytest.fixture(autouse=True)
def _isolate_from_ambient_env(monkeypatch):
    """Prevent ambient BELT_LLM_* env vars from leaking into tests.

    Tests that need specific env values set them explicitly via
    ``patch.dict`` or ``monkeypatch.setenv`` - those still work because
    monkeypatch only removes what was present at fixture setup time.

    Also opts the test session into the dotted-path escape hatches for
    agents and scorers so existing fixtures that reference
    ``tests.test_integration_flow.StubAgentAdapter`` or other in-test classes
    keep working. Tests that exercise the production default (escape
    hatch off) opt out via ``monkeypatch.delenv``.
    """
    for var in _BELT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("BELT_ALLOW_ARBITRARY_AGENT", "1")
    monkeypatch.setenv("BELT_ALLOW_ARBITRARY_SCORER", "1")
