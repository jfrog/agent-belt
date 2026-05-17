# (c) JFrog Ltd. (2026)

"""Parse newline-delimited JSON (NDJSON) streams.

Shared by agents that consume NDJSON CLI output.
"""

from __future__ import annotations

import json

from loguru import logger

# Defence-in-depth caps on top of the bounded stream reader in
# ``belt.agent.base.iter_bounded_stream``. Per-line size and
# parsed-JSON depth are both bounded so a malicious agent cannot smuggle
# pathological payloads past agents that bypass the bounded reader (e.g.
# unit tests, custom agents, or future entry-points).
_MAX_LINE_LEN = 1 * 1024 * 1024  # 1 MiB
_MAX_DEPTH = 64
_MAX_BOUNDED_BYTES = 50 * 1024 * 1024  # 50 MiB cap for bounded_json_loads


def bounded_json_loads(text: str, *, max_bytes: int = _MAX_BOUNDED_BYTES, max_depth: int = _MAX_DEPTH) -> object:
    """Parse a JSON document with explicit byte-size and nesting-depth caps.

    Raises :class:`ValueError` if ``text`` is too large or decodes into a
    structure nested deeper than ``max_depth``. Used outside the NDJSON
    pipeline (e.g. orchestrator workspace-state capture) where a hostile
    agent could otherwise smuggle a recursion bomb past unbounded
    ``json.loads``.
    """
    if len(text) > max_bytes:
        raise ValueError(f"json input exceeds max_bytes={max_bytes}")
    parsed = json.loads(text)
    if _max_depth(parsed) > max_depth:
        raise ValueError(f"json depth exceeds max_depth={max_depth}")
    return parsed


def _max_depth(value: object, current: int = 0) -> int:
    """Compute the maximum nesting depth of a JSON-decoded value."""
    if current > _MAX_DEPTH:
        return current
    if isinstance(value, dict):
        if not value:
            return current
        return max(_max_depth(v, current + 1) for v in value.values())
    if isinstance(value, list):
        if not value:
            return current
        return max(_max_depth(v, current + 1) for v in value)
    return current


def parse_ndjson(raw: str) -> list[dict]:
    """Parse newline-delimited JSON, warning on unparseable lines.

    Returns a list of parsed JSON objects. Lines that fail to parse, exceed
    :data:`_MAX_LINE_LEN`, or decode into objects nested deeper than
    :data:`_MAX_DEPTH` are logged at WARNING and skipped.
    """
    events: list[dict] = []
    skipped = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) > _MAX_LINE_LEN:
            skipped += 1
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if not isinstance(parsed, dict):
            skipped += 1
            continue
        if _max_depth(parsed) > _MAX_DEPTH:
            skipped += 1
            continue
        events.append(parsed)
    if skipped:
        logger.warning("NDJSON parse: skipped {} unparseable line(s) out of {}", skipped, skipped + len(events))
    return events
