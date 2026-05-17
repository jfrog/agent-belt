# (c) JFrog Ltd. (2026)

"""Pure helpers shared across exporters.

Per Design Principle 2, the entities in :mod:`belt.exporter.entities`
carry data only. Functions that *act* on those shapes (collapsing trial
expansions, normalising option types, capping byte sizes) live here so
exporter implementations stay focused on their output format.

Per-dimension iteration across scorers is delegated to
:func:`belt.scorer.payloads.iter_dimension_feedback`. Exporters call
that public helper rather than inlining "is this rules-shape or
llm-shape?" walks per output format - one source of truth for both
built-in and plugin-registered scorer payloads.
"""

from __future__ import annotations

from typing import Any, Iterable

from belt.constants import TRIAL_SUFFIX_RE
from belt.entities import ScenarioScore


def collapse_trials(scores: Iterable[ScenarioScore]) -> dict[str, list[ScenarioScore]]:
    """Group ``__trial_N``-suffixed scores back to their base scenario.

    The runner emits ``<scenario>__trial_K`` outcome dirs under ``--trials N``;
    the scorer mirrors that naming on disk. ``ExportContext.scores`` carries
    the expanded list. Exporters whose output format encodes one row / one
    record per scenario (e.g. JUnit) call this helper to re-group; exporters
    that surface every trial individually (e.g. JSONL streaming into a BI
    pipeline) iterate ``ExportContext.scores`` directly.

    Returns a dict keyed by ``"<group>/<base_scenario>"``. Order matches the
    first-seen order in ``scores`` for deterministic export output.
    """
    groups: dict[str, list[ScenarioScore]] = {}
    for s in scores:
        base = TRIAL_SUFFIX_RE.sub("", s.scenario_name)
        key = f"{s.group}/{base}"
        groups.setdefault(key, []).append(s)
    return groups


def get_int_option(options: dict[str, Any], key: str, default: int) -> int:
    """Read an integer option from a free-form ``--export-config`` block.

    YAML can hand the loader an ``int``, a ``str``, or - for misconfigured
    files - something exotic. Coerce defensively: accept ints directly,
    parse strings, fall back to ``default`` on anything else. Mirrors the
    permissiveness of ``envvars.get_int`` so behaviour is consistent across
    the framework.
    """
    raw = options.get(key, default)
    if isinstance(raw, bool):  # bool is an int subclass; reject explicitly
        return default
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            return default
    return default


def get_bool_option(options: dict[str, Any], key: str, default: bool) -> bool:
    """Read a boolean option from a free-form ``--export-config`` block."""
    raw = options.get(key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return default


def truncate_with_marker(text: str, max_bytes: int) -> str:
    """Truncate ``text`` to ``max_bytes`` of UTF-8, appending an explicit marker.

    JUnit consumers (Jenkins, GitLab, Buildkite, GitHub Actions test
    reporters) all parse XML eagerly into memory. A scenario whose error
    output is hundreds of MB of stdout will OOM the reporter, not just the
    eval run. Cap with a visible marker so the operator knows where to look
    on disk for the full picture instead of being silently fed the head.
    """
    if max_bytes <= 0:
        return ""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    head = encoded[:max_bytes].decode("utf-8", errors="replace")
    return head + "\n... [truncated; see run_dir for full output]"
