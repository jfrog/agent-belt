# (c) JFrog Ltd. (2026)

"""Doc-code parity test for the CLI.

``docs/glossary/CLI.md`` is intentionally a lean subcommand index and
workflow guide - the source of truth for individual flags is
``belt <subcommand> --help``. Maintaining a verbatim flag table in
markdown was the single largest source of doc rot in the legacy
``CLI.md``.

What we still enforce, mechanically:

1. Every subcommand exposed by the CLI is mentioned by name in
   ``CLI.md``. A new subcommand without a doc entry is a regression.
2. Every subcommand the doc references really exists on the CLI. Stops
   the doc from inventing or keeping stale subcommand names after a
   rename.

Flag-level parity is a reviewer concern, not a test concern: the
``--help`` output is always the truth.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_BELT_BIN = shutil.which("belt")

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_DOC = REPO_ROOT / "docs" / "glossary" / "CLI.md"

SUBCOMMANDS = [
    "eval",
    "run",
    "score",
    "aggregate",
    "export",
    "compare",
    "watch",
    "view",
    "doctor",
    "quickstart",
    "gc",
    "agent",
]

# Subcommand names that are plausible-sounding but have never existed on
# the CLI. The doc-references check uses this set so a typo or rename
# fails loudly rather than silently misleading readers.
_PLAUSIBLE_BUT_FAKE_SUBCOMMANDS = {
    "install",
    "init",
    "config",
    "serve",
    "publish",
    "build",
    "test",
}


def _doc_text() -> str:
    assert CLI_DOC.is_file(), f"missing doc file: {CLI_DOC}"
    return CLI_DOC.read_text()


def _help_text(argv: list[str]) -> str:
    """Run ``belt <argv> --help`` and return combined stdout+stderr."""
    if not _BELT_BIN:
        pytest.skip("belt console script not on PATH (install with `pip install -e .`)")
    proc = subprocess.run(
        [_BELT_BIN, *argv, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    text = proc.stdout + proc.stderr
    if not text.strip():
        raise RuntimeError(f"belt {' '.join(argv)} --help produced no output (rc={proc.returncode})")
    return text


@pytest.mark.parametrize("subcommand", SUBCOMMANDS)
def test_every_subcommand_is_indexed(subcommand: str) -> None:
    """Every ``belt <subcommand>`` is mentioned by name in CLI.md.

    Loose match: the literal token ``belt <subcommand>`` appears
    somewhere in the doc. A subcommand with no doc footprint is a
    discoverability bug - users browsing CLI.md will never know it
    exists.
    """
    doc = _doc_text()
    needle = f"belt {subcommand}"
    assert needle in doc, (
        f"Subcommand `{subcommand}` is exposed by the CLI but has no "
        f"mention in docs/glossary/CLI.md. Add it to the subcommand "
        f"index or to a workflow example."
    )


def test_subcommand_help_smoke() -> None:
    """Sanity: every subcommand actually accepts ``--help``.

    Catches a subcommand that was added to ``SUBCOMMANDS`` (and the
    doc) but is misregistered in the CLI dispatcher.
    """
    for sub in SUBCOMMANDS:
        # ``agent`` requires a sub-subcommand; help is exposed on the
        # parent group too, so we can call it directly.
        _help_text([sub])


def test_doc_referenced_subcommands_exist() -> None:
    """Every subcommand the docs talk about must really exist on the CLI."""
    doc = _doc_text()
    referenced = set(re.findall(r"belt\s+([a-z]+)\b", doc))
    real = set(SUBCOMMANDS)
    suspect = (referenced - real) & _PLAUSIBLE_BUT_FAKE_SUBCOMMANDS
    assert not suspect, f"docs/glossary/CLI.md references subcommands that do not exist on the CLI: {sorted(suspect)}"
