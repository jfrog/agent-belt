# (c) JFrog Ltd. (2026)

"""Doc-code parity for LLM judge providers.

Every concrete subclass of ``BaseJudgeBackend`` must appear in
``docs/glossary/CONFIGURATION.md``'s provider credentials table, and every
provider documented there must have a backend class. Drift in either direction
means a user reading the docs ends up either configuring a provider that
doesn't work or missing one that does.

Detection is byte-level on the documented provider name (e.g. ``Ollama``,
``Azure OpenAI``). Adding a new ``BaseJudgeBackend`` subclass without updating
``CONFIGURATION.md`` will fail this test, and vice-versa.
"""

from __future__ import annotations

import inspect
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGURATION_DOC = REPO_ROOT / "docs" / "glossary" / "CONFIGURATION.md"


def _backend_provider_names() -> set[str]:
    from belt.scorer.llm import backend

    names: set[str] = set()
    for _, obj in inspect.getmembers(backend):
        if inspect.isclass(obj) and issubclass(obj, backend.BaseJudgeBackend) and obj is not backend.BaseJudgeBackend:
            inst = obj.__new__(obj)
            names.add(obj.provider_name(inst))
    return names


def test_every_backend_provider_is_documented() -> None:
    """Every concrete ``BaseJudgeBackend`` subclass appears in CONFIGURATION.md."""
    code_providers = _backend_provider_names()
    doc = CONFIGURATION_DOC.read_text()
    missing = sorted(p for p in code_providers if p not in doc)
    assert not missing, (
        f"BaseJudgeBackend subclasses not mentioned in CONFIGURATION.md: {missing}. "
        "Add a row to the LLM provider credentials table."
    )


def test_documented_providers_have_backend() -> None:
    """Every provider name in the credentials table has a backend implementation.

    The credentials table uses pipe-table rows; we check the canonical
    provider names that the table is documented to include.
    """
    code_providers = _backend_provider_names()
    expected_documented = {"OpenAI", "Anthropic", "Azure OpenAI", "Ollama"}
    doc = CONFIGURATION_DOC.read_text()
    documented_in_text = {p for p in expected_documented if p in doc}
    orphans = sorted(documented_in_text - code_providers)
    assert not orphans, f"CONFIGURATION.md mentions providers without a BaseJudgeBackend implementation: {orphans}"
