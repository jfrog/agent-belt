# (c) JFrog Ltd. (2026)

"""Filesystem I/O helpers for the benchmark card.

Reading is best-effort: a missing or malformed input yields ``None``
rather than raising, so a partial run directory still produces a card
with empty optional fields. Writing logs failures but never raises -
the card is an artifact, not a precondition for run completion.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from belt._io import read_json
from belt.constants import BENCHMARK_CARD_JSON_FILE, BENCHMARK_CARD_MD_FILE, RESULTS_FILE

from .entities import BenchmarkCard

# Re-exported so ``from belt.benchmark_card.io import read_json``
# (used by :mod:`belt.benchmark_card.collect`) keeps working
# unchanged. The single implementation now lives in
# :mod:`belt._io` so the runner / commands / scorer can share it
# without crossing the aggregator-phase boundary.
__all__ = [
    "iso_utc",
    "load_results_for_card",
    "read_json",
    "started_at_from_run_dir",
    "write_card",
]


def iso_utc(dt: datetime | None = None) -> str:
    """Return an ISO-8601 UTC timestamp with second precision and ``Z`` suffix."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def started_at_from_run_dir(run_dir: Path) -> str:
    """Best-effort fallback timestamp for a run with no recorded ``started_at``.

    The canonical source is the ``started_at`` field that
    ``commands/run.py`` writes into ``run_meta.json`` at run-init time
    (a real UTC timestamp). This helper only runs when that field is
    missing - typically when reading a run directory produced by an
    older belt, or one constructed manually for tests.

    Run directories are named ``YYYYMMDD-HHMMSS-<hex>`` from
    :func:`datetime.now()` (local time, no timezone info), so the name
    cannot be safely promoted to UTC. The directory's ``mtime`` is a
    correct upper bound on the actual start time, so we use that. As a
    last resort (e.g. the directory was deleted between check and read)
    we return the current time.
    """
    try:
        return iso_utc(datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc))
    except OSError:
        return iso_utc()


def write_card(card: BenchmarkCard, run_dir: Path) -> tuple[Path, Path]:
    """Persist the card as JSON and Markdown side-by-side in ``run_dir``.

    Returns ``(json_path, md_path)``. Failures are logged but not raised:
    the card is a best-effort artifact, never a precondition for run
    completion.
    """
    from .render import render_markdown

    json_path = run_dir / BENCHMARK_CARD_JSON_FILE
    md_path = run_dir / BENCHMARK_CARD_MD_FILE
    # ``card.model_dump_json`` already produces canonical JSON, so we
    # bypass :func:`belt._io.write_json` (which would re-serialise
    # the model_dump dict and lose Pydantic's field ordering /
    # ``mode="json"`` coercions).
    try:
        json_path.write_text(card.model_dump_json(indent=2) + "\n")
    except OSError as e:
        logger.warning("Failed to write {}: {}", BENCHMARK_CARD_JSON_FILE, e)
    try:
        md_path.write_text(render_markdown(card))
    except OSError as e:
        logger.warning("Failed to write {}: {}", BENCHMARK_CARD_MD_FILE, e)
    return json_path, md_path


def load_results_for_card(run_dir: Path) -> dict[str, Any]:
    """Read ``results.json`` for ad-hoc card construction outside ``aggregate``.

    Returns an empty dict if the file is missing or malformed; callers
    that care about completeness should check the result. Used by tests
    and any external tool that wants to regenerate a card from a
    finished run dir.
    """
    return read_json(run_dir / RESULTS_FILE) or {}
