# (c) JFrog Ltd. (2026)

"""Canonical JSON read/write helpers.

The belt pipeline persists hundreds of small JSON files per run -
``run_meta.json``, per-scenario ``score.json`` / ``_runtime_info.json``
sidecars, the manifest, the benchmark card. They all want the same
two things:

1. **Best-effort reads.** Missing or malformed input must not poison
   the surrounding feature; the caller falls back to "no data" and
   continues. Failures are debug-logged so an operator can find them
   without the user seeing a stack trace at the end of every run.
2. **Best-effort writes.** Persistence is an artefact, not a
   precondition for the surrounding operation succeeding. Writes log
   failures and return a boolean rather than raising.

This module concentrates both behaviours behind two helpers so a new
caller does not need to re-decide how to spell ``json.dumps(..., indent=2,
sort_keys=True) + "\\n"`` and how to wrap the ``OSError`` /
``json.JSONDecodeError`` swallowing. ``benchmark_card.io.read_json``
re-exports :func:`read_json` from here for module-local imports inside
the aggregator phase.

Stricter writers (atomic temp-file + rename, fsync, file locking) live
in their callers - the manifest writer in particular needs atomicity
because it underpins crash-resume and orphan cleanup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger


def read_json(path: Path | str) -> dict[str, Any] | None:
    """Read and parse a JSON file; return ``None`` on missing or malformed input.

    The canonical behaviour for every "best-effort" reader in
    belt. Used by:

    - the benchmark-card collector (``run_meta``, scenario sidecars,
      results), where a missing file from one scenario must not poison
      the entire card;
    - the run / aggregate commands when surfacing prior-run state;
    - any future caller that wants "load this file if it exists, else
      pretend it didn't".

    Errors are debug-logged (operator can find them) but never raise.
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Failed to read {}: {}", p, e)
        return None


def write_json(
    path: Path | str,
    data: Any,
    *,
    indent: int = 2,
    sort_keys: bool = True,
) -> bool:
    """Persist ``data`` as pretty JSON with a trailing newline.

    Returns ``True`` if the file was written, ``False`` on any
    :class:`OSError` (failure is debug-logged). The trailing newline
    keeps tools that expect POSIX text files (``cat``, ``less``,
    pre-commit hooks) happy.

    Defaults match the project convention (``indent=2,
    sort_keys=True``) so persisted card / sidecar files diff cleanly
    across runs. Callers that need NDJSON or jsonl streams should not
    use this helper.
    """
    p = Path(path)
    try:
        p.write_text(json.dumps(data, indent=indent, sort_keys=sort_keys) + "\n")
        return True
    except OSError as e:
        logger.debug("Failed to write {}: {}", p, e)
        return False


__all__ = ["read_json", "write_json"]
