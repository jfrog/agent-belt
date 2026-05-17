# (c) JFrog Ltd. (2026)

"""Doc-code parity for the exporter surface.

Mirrors :mod:`tests.test_agent_doc_parity` and :mod:`tests.test_provider_doc_parity`
- every built-in exporter must appear by name in
``docs/glossary/PLUGGABILITY.md`` (the authoring guide), so the
user-facing documentation never silently drifts behind the registry.
A new built-in that ships without documentation fails this test.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PLUGGABILITY = _REPO_ROOT / "docs" / "glossary" / "PLUGGABILITY.md"


def test_every_builtin_documented():
    from belt.exporter.registry import _EXPORTER_REGISTRY

    text = _PLUGGABILITY.read_text(encoding="utf-8")
    missing = [name for name in _EXPORTER_REGISTRY if name not in text]
    assert not missing, (
        f"Built-in exporter(s) not documented in PLUGGABILITY.md: {missing}. "
        f"Add a mention to the exporter authoring section."
    )


def test_pluggability_doc_exists():
    """Plugin authoring guide ships at the consolidated location."""
    assert _PLUGGABILITY.is_file(), f"Missing {_PLUGGABILITY.relative_to(_REPO_ROOT)} - the plugin authoring guide."


def test_pluggability_mentions_exporter_contracts():
    text = _PLUGGABILITY.read_text(encoding="utf-8")
    for token in ("BaseExporter", "belt.exporters", "ExportContext"):
        assert token in text, f"PLUGGABILITY.md must reference {token}"
